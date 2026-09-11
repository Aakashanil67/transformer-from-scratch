"""Atomic training checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    step: int,
    history: dict[str, Any] | None = None,
    generators: dict[str, torch.Generator] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save all state needed to resume a local run."""
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "history": history or {},
        "config": config or {},
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if generators:
        payload["generator_states"] = {
            name: generator.get_state() for name, generator in generators.items()
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    generators: dict[str, torch.Generator] | None = None,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint and return its metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_config is not None and payload.get("config", {}) != expected_config:
        raise ValueError("checkpoint configuration does not match the requested run")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    for name, state in payload.get("generator_states", {}).items():
        if generators and name in generators:
            generators[name].set_state(state)
    return payload
