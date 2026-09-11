import pytest
import torch

from transformer_lab.data.batches import sample_batch


def test_sample_batch_returns_shifted_fixed_length_sequences() -> None:
    token_ids = torch.arange(20)
    generator = torch.Generator().manual_seed(4)

    inputs, targets = sample_batch(
        token_ids,
        batch_size=3,
        block_size=4,
        generator=generator,
    )

    assert inputs.shape == (3, 4)
    assert torch.equal(targets, inputs + 1)


def test_sample_batch_rejects_a_sequence_that_is_too_short() -> None:
    with pytest.raises(ValueError, match=r"at least block_size \+ 1 tokens"):
        sample_batch(torch.arange(4), batch_size=1, block_size=4)
