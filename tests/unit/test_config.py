import pytest

from transformer_lab.config import GPTConfig


def test_rejects_embedding_width_that_cannot_be_split_across_heads() -> None:
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=3, n_embd=10)


def test_rejects_non_positive_vocabulary_size() -> None:
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        GPTConfig(vocab_size=0, block_size=8, n_layer=1, n_head=2, n_embd=8)


def test_rejects_non_positive_head_count() -> None:
    with pytest.raises(ValueError, match="n_head must be positive"):
        GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=0, n_embd=8)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"block_size": 0}, "block_size must be positive"),
        ({"n_layer": 0}, "n_layer must be positive"),
        ({"n_embd": 0}, "n_embd must be positive"),
        ({"dropout": 1.0}, "dropout must be in the interval"),
    ],
)
def test_rejects_invalid_model_dimensions(overrides: dict[str, int | float], message: str) -> None:
    values: dict[str, int | float] = {
        "vocab_size": 32,
        "block_size": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        GPTConfig(**values)
