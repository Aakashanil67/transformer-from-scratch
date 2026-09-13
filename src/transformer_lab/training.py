"""Deterministic, resumable training utilities for autoregressive models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from transformer_lab.checkpointing import load_checkpoint, save_checkpoint
from transformer_lab.data.batches import sample_batch


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    batch_size: int
    block_size: int
    learning_rate: float
    eval_interval: int
    eval_batches: int
    seed: int
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    amp: bool = False

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "block_size", "eval_interval", "eval_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        update_budget = math.ceil(self.steps / self.gradient_accumulation_steps)
        if self.warmup_steps > update_budget:
            raise ValueError("warmup_steps cannot exceed the optimiser update budget")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass
class TrainingHistory:
    """Step-indexed losses and throughput measurements."""

    validation_losses: list[float] = field(default_factory=list)
    training_losses: list[float] = field(default_factory=list)
    validation_steps: list[int] = field(default_factory=list)
    tokens_seen: int = 0
    optimizer_steps: int = 0
    elapsed_seconds: float = 0.0


def _estimate_loss(
    model: nn.Module,
    token_ids: Tensor,
    *,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    losses: list[Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(config.eval_batches):
            inputs, targets = sample_batch(
                token_ids,
                batch_size=config.batch_size,
                block_size=config.block_size,
                generator=generator,
            )
            _, loss = model(inputs.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("language model did not return a training loss")
            losses.append(loss.detach())
    model.train(was_training)
    return torch.stack(losses).mean().item()


def _generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"language-model sampling supports CPU and CUDA tensors; received {device.type}"
        )
    generator = torch.Generator(device=device.type)
    return generator.manual_seed(seed)


def _lr_lambda(step: int, config: TrainingConfig) -> float:
    update_budget = math.ceil(config.steps / config.gradient_accumulation_steps)
    if config.warmup_steps and step <= config.warmup_steps:
        return step / config.warmup_steps
    remaining = max(update_budget - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / remaining, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_language_model(
    model: nn.Module,
    *,
    train_token_ids: Tensor,
    validation_token_ids: Tensor,
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    checkpoint_config: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> TrainingHistory:
    """Train a model with independent sampling streams and optional resume support."""
    model_device = next(model.parameters()).device
    train_generator = _generator(config.seed, train_token_ids.device)
    eval_generator = _generator(config.seed + 1, validation_token_ids.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_lambda(step, config))
    amp_enabled = bool(config.amp and model_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = TrainingHistory()
    start_step = 0
    generators = {"train": train_generator, "eval": eval_generator}
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("resume requested but checkpoint_path does not exist")
        payload = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            generators=generators,
            expected_config=checkpoint_config or {},
        )
        start_step = int(payload["step"])
        if start_step < 0 or start_step > config.steps:
            raise ValueError("checkpoint microstep is outside the requested training budget")
        if start_step % config.gradient_accumulation_steps != 0 and start_step != config.steps:
            raise ValueError(
                "checkpoint was saved during an incomplete accumulation window and "
                "cannot be resumed"
            )
        saved_history = payload.get("history", {})
        history.validation_losses = list(saved_history.get("validation_losses", []))
        history.training_losses = list(saved_history.get("training_losses", []))
        history.validation_steps = list(saved_history.get("validation_steps", []))
        history.tokens_seen = int(saved_history.get("tokens_seen", 0))
        history.optimizer_steps = int(
            payload.get("optimizer_step", saved_history.get("optimizer_steps", start_step))
        )
        history.elapsed_seconds = float(saved_history.get("elapsed_seconds", 0.0))
    else:
        history.validation_losses.append(
            _estimate_loss(
                model,
                validation_token_ids,
                config=config,
                generator=eval_generator,
                device=model_device,
            )
        )
        history.validation_steps.append(0)

    previous_elapsed = history.elapsed_seconds
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step in range(start_step + 1, config.steps + 1):
        window_start = ((step - 1) // config.gradient_accumulation_steps) * (
            config.gradient_accumulation_steps
        ) + 1
        window_size = min(config.gradient_accumulation_steps, config.steps - window_start + 1)
        inputs, targets = sample_batch(
            train_token_ids,
            batch_size=config.batch_size,
            block_size=config.block_size,
            generator=train_generator,
        )
        with torch.amp.autocast(device_type=model_device.type, enabled=amp_enabled):
            _, loss = model(inputs.to(model_device), targets.to(model_device))
            if loss is None:
                raise RuntimeError("language model did not return a training loss")
            scaled_loss = loss / window_size
        scaler.scale(scaled_loss).backward()
        completed_update = False
        if step % config.gradient_accumulation_steps == 0 or step == config.steps:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            history.optimizer_steps += 1
            completed_update = True
        history.training_losses.append(float(loss.detach().cpu()))
        history.tokens_seen += config.batch_size * config.block_size

        if step % config.eval_interval == 0 or step == config.steps:
            history.validation_losses.append(
                _estimate_loss(
                    model,
                    validation_token_ids,
                    config=config,
                    generator=eval_generator,
                    device=model_device,
                )
            )
            history.validation_steps.append(step)
        if checkpoint_path is not None and completed_update:
            history.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                optimizer_step=history.optimizer_steps,
                history={
                    "validation_losses": history.validation_losses,
                    "training_losses": history.training_losses,
                    "validation_steps": history.validation_steps,
                    "tokens_seen": history.tokens_seen,
                    "optimizer_steps": history.optimizer_steps,
                    "elapsed_seconds": history.elapsed_seconds,
                },
                generators=generators,
                config=checkpoint_config,
                metadata=checkpoint_metadata,
            )
    history.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
    return history
