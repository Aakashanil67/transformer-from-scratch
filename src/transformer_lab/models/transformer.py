"""Decoder-only transformer components built from PyTorch primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from transformer_lab.config import GPTConfig


def transformer_parameter_count(config: GPTConfig) -> int:
    """Return the exact count for the tied, GPT-style decoder implementation."""
    embedding_parameters = (config.vocab_size + config.block_size) * config.n_embd
    if config.bias:
        block_parameters = 12 * config.n_embd**2 + 13 * config.n_embd
        final_layer_norm = 2 * config.n_embd
    else:
        block_parameters = 12 * config.n_embd**2 + 2 * config.n_embd
        final_layer_norm = config.n_embd
    return embedding_parameters + config.n_layer * block_parameters + final_layer_norm


def attention_lora_parameter_count(config: GPTConfig, *, rank: int) -> int:
    """Count rank-r adapters on every attention input/output projection."""
    if isinstance(rank, bool) or rank <= 0:
        raise ValueError("rank must be positive")
    return (
        config.n_layer * rank * (config.n_embd + 3 * config.n_embd + config.n_embd + config.n_embd)
    )


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a lower-triangular attention mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply masked attention to tensors shaped ``(batch, sequence, embedding)``."""
        _, sequence_length, embedding_size = inputs.shape
        queries, keys, values = self.c_attn(inputs).split(embedding_size, dim=2)
        queries = queries.view(-1, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        keys = keys.view(-1, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        values = values.view(-1, sequence_length, self.n_head, self.head_size).transpose(1, 2)

        weights = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        weights = weights.masked_fill(
            ~self.causal_mask[:, :, :sequence_length, :sequence_length], -torch.inf
        )
        weights = self.attention_dropout(F.softmax(weights, dim=-1))
        output = weights @ values
        output = output.transpose(1, 2).contiguous().view(-1, sequence_length, embedding_size)
        return self.residual_dropout(self.c_proj(output))


class MLP(nn.Module):
    """The feed-forward sublayer used after self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.c_proj(self.activation(self.c_fc(inputs))))


class TransformerBlock(nn.Module):
    """A pre-normalization decoder block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.ln_1(inputs))
        return inputs + self.mlp(self.ln_2(inputs))


class DecoderOnlyTransformer(nn.Module):
    """A GPT-style autoregressive language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return token logits and optional next-token cross-entropy loss."""
        hidden_states = self.hidden_states(token_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("target shape must match token_ids shape")
            if targets.dtype not in (torch.int32, torch.int64):
                raise ValueError("targets must contain integer token IDs")
            if torch.any(targets < 0) or torch.any(targets >= self.config.vocab_size):
                raise ValueError("targets contain an out-of-range token ID")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def _validate_token_ids(self, token_ids: Tensor) -> int:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must contain integer token IDs")
        _, sequence_length = token_ids.shape
        if sequence_length == 0:
            raise ValueError("token_ids sequence must not be empty")
        if sequence_length > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.config.vocab_size):
            raise ValueError("token_ids contain an out-of-range token ID")
        return sequence_length

    def hidden_states(self, token_ids: Tensor) -> Tensor:
        """Return the final hidden state before the language-model head."""
        return self.layer_states(token_ids)[-1]

    def layer_states(self, token_ids: Tensor) -> tuple[Tensor, ...]:
        """Expose GPT-2-compatible hidden-state boundaries for parity checks."""
        sequence_length = self._validate_token_ids(token_ids)
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden_states = self.dropout(hidden_states)
        states = [hidden_states]
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if index < len(self.blocks) - 1:
                states.append(hidden_states)
        states.append(self.ln_f(hidden_states))
        return tuple(states)

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample tokens while retaining at most one model context window."""
        self._validate_token_ids(token_ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and (top_k <= 0 or top_k > self.config.vocab_size):
            raise ValueError("top_k must be between 1 and vocab_size")
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], -torch.inf)
            if do_sample:
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
