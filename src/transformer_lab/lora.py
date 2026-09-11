"""Low-rank adapters for frozen linear projections."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAConfig:
    """Settings shared by every injected adapter in a model."""

    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")


@dataclass(frozen=True)
class InjectionReport:
    """Names and parameter counts produced by adapter injection."""

    replaced_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


@dataclass(frozen=True)
class ParameterReport:
    """Trainable and total parameter counts."""

    trainable: int
    total: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable rank-constrained update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.a), self.b)
        return self.base(inputs) + self.scale * update

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield self.a
        yield self.b


def parameter_report(model: nn.Module) -> ParameterReport:
    """Count all parameters, including frozen base weights."""
    parameters = list(model.parameters())
    return ParameterReport(
        trainable=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        total=sum(parameter.numel() for parameter in parameters),
    )


def _matches(name: str, targets: Sequence[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def inject_lora(model: nn.Module, config: LoRAConfig) -> InjectionReport:
    """Replace configured linear modules with zero-initialised adapters."""
    model.requires_grad_(False)
    replaced: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _matches(full_name, config.target_modules):
                setattr(
                    parent,
                    child_name,
                    LoRALinear(
                        child,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    ),
                )
                replaced.append(full_name)
    if not replaced:
        targets = ", ".join(config.target_modules)
        raise ValueError(f"no linear modules matched target_modules: {targets}")
    counts = parameter_report(model)
    return InjectionReport(tuple(replaced), counts.trainable, counts.total)


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return only adapter tensors, excluding frozen base weights."""
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.endswith(".a") or name.endswith(".b")
    }


def load_adapter_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Load adapter tensors and reject missing or unexpected adapter keys."""
    expected = set(adapter_state_dict(model))
    received = set(state)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    if missing or unexpected:
        raise ValueError(f"adapter keys differ; missing={missing}, unexpected={unexpected}")
    current = model.state_dict()
    for name, tensor in state.items():
        if current[name].shape != tensor.shape:
            raise ValueError(f"adapter shape mismatch for {name}")
        current[name].copy_(tensor)
