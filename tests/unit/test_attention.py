import torch

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import CausalSelfAttention


def test_causal_self_attention_preserves_batch_sequence_and_embedding_shape() -> None:
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=6,
    )
    attention = CausalSelfAttention(config)

    output = attention(torch.randn(3, 5, 6))

    assert output.shape == (3, 5, 6)


def test_causal_self_attention_does_not_read_future_positions() -> None:
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=6,
    )
    attention = CausalSelfAttention(config).eval()
    inputs = torch.randn(1, 5, 6)
    changed_inputs = inputs.clone()
    changed_inputs[:, -1, :] += 100.0

    original = attention(inputs)
    changed = attention(changed_inputs)

    assert torch.allclose(original[:, :-1], changed[:, :-1])
