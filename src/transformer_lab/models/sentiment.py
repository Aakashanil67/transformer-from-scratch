"""A small classification head over the decoder's final hidden state."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import DecoderOnlyTransformer


class SentimentClassifier(nn.Module):
    """Pool the final non-padding token and predict a sentiment class."""

    def __init__(self, config: GPTConfig, *, num_labels: int) -> None:
        super().__init__()
        if num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        self.backbone = DecoderOnlyTransformer(config)
        self.classifier = nn.Linear(config.n_embd, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return class logits and an optional cross-entropy loss."""
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids shape")
            if attention_mask.dtype not in (torch.int32, torch.int64, torch.bool):
                raise ValueError("attention_mask must be boolean or integer")
        hidden = self.backbone.hidden_states(input_ids)
        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            lengths = attention_mask.to(torch.long).sum(dim=1)
            if torch.any(lengths <= 0):
                raise ValueError("attention_mask must contain at least one valid token")
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths - 1]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if labels.ndim != 1 or labels.size(0) != input_ids.size(0):
                raise ValueError("labels must have shape (batch,)")
            loss = F.cross_entropy(logits, labels)
        return logits, loss
