"""Serialization boundary for low-rank adapter weights."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from transformer_lab.lora import (
    InjectionReport,
    LoRAConfig,
    LoRALinear,
    ParameterReport,
    adapter_state_dict,
    inject_lora,
    load_adapter_state_dict,
    parameter_report,
)


def save_adapter(path: Path, model: nn.Module, config: LoRAConfig) -> Path:
    payload = {
        "schema_version": 1,
        "config": asdict(config),
        "state_dict": adapter_state_dict(model),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_adapter(path: Path, model: nn.Module) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"adapter artifact does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported adapter artifact schema")
    load_adapter_state_dict(model, payload["state_dict"])
    return dict(payload["config"])


__all__ = [
    "InjectionReport",
    "LoRAConfig",
    "LoRALinear",
    "ParameterReport",
    "adapter_state_dict",
    "inject_lora",
    "load_adapter",
    "load_adapter_state_dict",
    "parameter_report",
    "save_adapter",
]
