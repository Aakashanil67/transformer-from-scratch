"""A minimal next-token baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict each token from the token immediately before it."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.token_logits(token_ids)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

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
        """Generate tokens autoregressively from the final token in the sequence."""
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError(
                "token_ids must have shape (batch, sequence) with a non-empty sequence"
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and (top_k <= 0 or top_k > self.token_logits.num_embeddings):
            raise ValueError("top_k must be between 1 and vocab_size")
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids[:, -1:])
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
