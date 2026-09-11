"""Random contiguous batches for next-token prediction."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_batch(
    token_ids: Tensor,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample input sequences and their one-token-shifted targets."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if len(token_ids) < block_size + 1:
        raise ValueError("token_ids must contain at least block_size + 1 tokens")

    starts = torch.randint(
        len(token_ids) - block_size,
        (batch_size,),
        generator=generator,
        device=token_ids.device,
    )
    inputs = torch.stack([token_ids[start : start + block_size] for start in starts])
    targets = torch.stack([token_ids[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets
