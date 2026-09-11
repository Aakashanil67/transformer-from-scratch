import pytest
import torch

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import DecoderOnlyTransformer


def test_seeded_top_k_sampling_is_repeatable() -> None:
    model = DecoderOnlyTransformer(
        GPTConfig(vocab_size=12, block_size=8, n_layer=1, n_head=2, n_embd=8)
    ).eval()
    prompt = torch.tensor([[1, 2]])
    first = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.7,
        top_k=4,
        generator=torch.Generator().manual_seed(4),
    )
    second = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.7,
        top_k=4,
        generator=torch.Generator().manual_seed(4),
    )

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": 1, "temperature": 0}, "temperature"),
        ({"max_new_tokens": 1, "top_k": 0}, "top_k"),
    ],
)
def test_generation_rejects_invalid_sampling_controls(kwargs, message: str) -> None:
    model = DecoderOnlyTransformer(
        GPTConfig(vocab_size=12, block_size=8, n_layer=1, n_head=2, n_embd=8)
    )

    with pytest.raises(ValueError, match=message):
        model.generate(torch.tensor([[1, 2]]), **kwargs)
