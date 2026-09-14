"""Shared training and evaluation workflow for Financial PhraseBank."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from joblib import load
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lab.checkpointing import load_checkpoint, save_checkpoint
from transformer_lab.config import GPTConfig
from transformer_lab.config_io import ExperimentConfig, effective_config
from transformer_lab.data.financial_phrasebank import (
    evaluation_example_ids,
    load_phrasebank,
    split_phrasebank,
    split_summary,
)
from transformer_lab.evaluation.classification import (
    apply_temperature,
    classification_metrics,
    fit_temperature,
    multiclass_log_loss,
)
from transformer_lab.experiments.manifests import build_manifest, config_digest, resolve_run
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import (
    environment_metadata,
    file_sha256,
    peak_memory,
    source_provenance,
)
from transformer_lab.gpt2 import REVISION, gpt2_small_config, load_huggingface_gpt2_weights
from transformer_lab.lora import LoRAConfig, inject_lora, parameter_report
from transformer_lab.models.sentiment import SentimentClassifier

REPOSITORY = "financial_phrasebank"
ARCHIVE = "data/FinancialPhraseBank-v1.0.zip"
ARCHIVE_MEMBER = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
LABELS = ["negative", "neutral", "positive"]


def sentiment_seeds(config: ExperimentConfig, seeds: list[int]) -> list[ExperimentConfig]:
    """Derive isolated run paths while leaving the dataset partition unchanged."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list without duplicates")
    return [resolve_run(config, seed=seed) for seed in seeds]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unavailable(config: ExperimentConfig, device: torch.device, message: str) -> Path:
    provenance = source_provenance(config.source_path)
    record = ExperimentRecord(
        run_id=config.experiment["name"],
        status="unavailable",
        model={"mode": config.model.get("mode")},
        config=effective_config(config),
        environment=environment_metadata(device),
        provenance=provenance,
        manifest=build_manifest(config, provenance=provenance),
        metrics=None,
        error={"type": "PreflightError", "message": message},
        schema_version=3,
    )
    result_path = Path(config.output["result"])
    record.write(result_path)
    return result_path


def _encode(tokenizer: Any, examples: list[Any], max_length: int) -> tuple[Tensor, Tensor, Tensor]:
    encoded = tokenizer(
        [example.text for example in examples],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([LABELS.index(example.label) for example in examples], dtype=torch.long)
    return encoded["input_ids"], encoded["attention_mask"], labels


def _training_partitions(
    splits: Any, *, validation_only: bool
) -> tuple[Sequence[Any], Sequence[Any], Sequence[Any] | None]:
    """Return only the partitions needed by a training/evaluation mode."""
    return splits.train, splits.validation, None if validation_only else splits.test


def _predict(
    model: SentimentClassifier,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()
    targets: list[Tensor] = []
    predictions: list[Tensor] = []
    probabilities: list[Tensor] = []
    raw_logits: list[Tensor] = []
    try:
        with torch.no_grad():
            for input_ids, attention_mask, labels in loader:
                logits, _ = model(
                    input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                targets.append(labels)
                predictions.append(logits.argmax(dim=-1).cpu())
                probabilities.append(logits.softmax(dim=-1).cpu())
                raw_logits.append(logits.cpu())
        return (
            torch.cat(targets).numpy(),
            torch.cat(predictions).numpy(),
            torch.cat(probabilities).numpy(),
            torch.cat(raw_logits).numpy(),
        )
    finally:
        model.train(was_training)


def _train_transformer(
    model: SentimentClassifier,
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    validation_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    accumulation: int,
    max_grad_norm: float,
    amp: bool,
    checkpoint_path: Path | None = None,
    best_model_path: Path | None = None,
    checkpoint_config: dict[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses: list[float] = []
    candidates: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = 0
    best_score = -1.0
    elapsed_before = 0.0
    generators = {"train": train_loader.generator} if train_loader.generator else {}
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("resume requested but the sentiment checkpoint is missing")
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            generators=generators,
            expected_config=checkpoint_config or {},
        )
        start_epoch = int(checkpoint["step"])
        history = checkpoint.get("history", {})
        losses = list(history.get("losses", []))
        candidates = list(history.get("validation_candidates", []))
        best_epoch = int(history.get("best_epoch", 0))
        best_score = float(history.get("best_score", -1.0))
        elapsed_before = float(history.get("train_seconds", 0.0))
    started = time.perf_counter()
    model.train()
    for epoch in range(start_epoch + 1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
            remainder = len(train_loader) % accumulation
            divisor = (
                remainder if remainder and step > len(train_loader) - remainder else accumulation
            )
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                _, loss = model(
                    input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )
                if loss is None:
                    raise RuntimeError("sentiment model did not return a loss")
                scaled_loss = loss / divisor
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        targets, predictions, probabilities, raw_logits = _predict(model, validation_loader, device)
        validation = classification_metrics(
            targets,
            predictions,
            labels=list(range(len(LABELS))),
            probabilities=probabilities,
            bootstrap_samples=100,
        )
        validation["loss"] = multiclass_log_loss(
            targets,
            apply_temperature(raw_logits, 1.0),
            labels=list(range(len(LABELS))),
        )
        candidates.append({"epoch": epoch, "metrics": validation})
        score = float(validation["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            if best_model_path is not None:
                _atomic_torch_save({"state_dict": model.state_dict()}, best_model_path)
        if checkpoint_path is not None:
            elapsed = elapsed_before + time.perf_counter() - started
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=epoch,
                history={
                    "losses": losses,
                    "validation_candidates": candidates,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "train_seconds": elapsed,
                },
                generators=generators,
                config=checkpoint_config,
            )
        model.train()
    if best_model_path is not None:
        if not best_model_path.exists():
            raise FileNotFoundError("selected best sentiment checkpoint is missing")
        try:
            best = torch.load(best_model_path, map_location=device, weights_only=True)
            model.load_state_dict(best["state_dict"])
        except (OSError, KeyError, RuntimeError, TypeError) as error:
            raise ValueError("selected best sentiment checkpoint is corrupt") from error
    best_validation = next(
        candidate["metrics"] for candidate in candidates if candidate["epoch"] == best_epoch
    )
    return {
        "losses": losses,
        "validation": best_validation,
        "validation_candidates": candidates,
        "best_epoch": best_epoch,
        "train_seconds": elapsed_before + time.perf_counter() - started,
    }


def _write_validation_summary(
    path: Path,
    *,
    config: ExperimentConfig,
    method: str,
    validation: dict[str, Any],
    validation_candidates: list[dict[str, Any]],
    train_seconds: float,
    data: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    status: str = "completed",
    error: dict[str, Any] | None = None,
    candidate_id: str | None = None,
    identity_config: ExperimentConfig | None = None,
) -> Path:
    """Persist a candidate record whose payload contains validation data only."""
    identity = identity_config or config
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sentiment-candidate",
        "candidate_id": candidate_id or str(config.experiment["name"]),
        "method": method,
        "status": status,
        "effective_config": effective_config(identity),
        "config_digest": config_digest(identity),
        "data": data,
        "timing": {"train_seconds": train_seconds},
    }
    if status == "completed":
        payload.update(
            {
                "validation": validation,
                "validation_candidates": validation_candidates,
                "artifacts": artifacts or {},
            }
        )
    if error is not None:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def _checkpoint_payload(
    model: SentimentClassifier,
    *,
    mode: str,
    revision: str,
    config: ExperimentConfig,
    max_length: int,
    calibration_temperature: float = 1.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "state_dict": model.state_dict(),
        "architecture": asdict(model.backbone.config),
        "mode": mode,
        "revision": revision,
        "tokenizer_id": config.model.get("base_model", "openai-community/gpt2"),
        "tokenizer_revision": revision,
        "max_length": max_length,
        "labels": LABELS,
        "num_labels": len(LABELS),
        "calibration_temperature": calibration_temperature,
    }
    if mode == "lora":
        payload["lora"] = {
            "rank": int(config.model.get("lora_rank", 4)),
            "alpha": float(config.model.get("lora_alpha", 8)),
            "dropout": float(config.model.get("lora_dropout", 0)),
            "target_modules": tuple(
                config.model.get("target_modules", ["attention.c_attn", "attention.c_proj"])
            ),
        }
    return payload


def run_sentiment_experiment(
    config: ExperimentConfig,
    *,
    device: torch.device,
    seed: int | None = None,
    resume: bool = False,
) -> Path:
    """Run one sentiment mode and retain a diagnostic record after a CUDA OOM."""
    config = resolve_run(config, seed=seed)
    try:
        return _execute_sentiment_experiment(config, device=device, resume=resume)
    except torch.OutOfMemoryError as error:
        if device.type != "cuda":
            raise
        provenance = source_provenance(config.source_path)
        record = ExperimentRecord(
            run_id=str(config.experiment["name"]),
            status="failed",
            config=effective_config(config),
            model={"mode": config.model.get("mode")},
            environment=environment_metadata(device),
            provenance=provenance,
            manifest=build_manifest(config, provenance=provenance),
            memory=peak_memory(device),
            metrics=None,
            error={"type": type(error).__name__, "message": str(error)},
            schema_version=3,
        )
        result_path = Path(config.output["result"])
        record.write(result_path)
        torch.cuda.empty_cache()
        return result_path


def run_sentiment_candidate(
    config: ExperimentConfig,
    *,
    device: torch.device,
    summary_path: Path,
    seed: int | None = None,
    resume: bool = False,
    candidate_id: str | None = None,
    identity_config: ExperimentConfig | None = None,
) -> Path:
    """Train one candidate and write a validation-only summary."""
    config = resolve_run(config, seed=seed)
    mode = str(config.model.get("mode", "lora"))
    if mode == "baseline":
        from transformer_lab.experiments.baseline import run_baseline

        return run_baseline(
            config,
            validation_only=True,
            summary_path=summary_path,
            identity_config=identity_config,
            candidate_id=candidate_id,
        )
    if device.type != "cuda":
        return _write_validation_summary(
            summary_path,
            config=config,
            method=mode,
            validation={},
            validation_candidates=[],
            train_seconds=0.0,
            data={"train_rows": None, "validation_rows": None},
            status="unavailable",
            error={
                "type": "PreflightError",
                "message": "GPT-2 candidate training requires CUDA for this protocol",
            },
            candidate_id=candidate_id,
            identity_config=identity_config,
        )
    try:
        return _execute_sentiment_experiment(
            config,
            device=device,
            seed=seed,
            resume=resume,
            validation_only=True,
            validation_summary_path=summary_path,
            candidate_id=candidate_id,
            identity_config=identity_config,
        )
    except torch.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        return _write_validation_summary(
            summary_path,
            config=config,
            method=mode,
            validation={},
            validation_candidates=[],
            train_seconds=0.0,
            data={"train_rows": None, "validation_rows": None},
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
            candidate_id=candidate_id,
            identity_config=identity_config,
        )


def _execute_sentiment_experiment(
    config: ExperimentConfig,
    *,
    device: torch.device,
    seed: int | None = None,
    resume: bool = False,
    validation_only: bool = False,
    validation_summary_path: Path | None = None,
    candidate_id: str | None = None,
    identity_config: ExperimentConfig | None = None,
) -> Path:
    """Execute one transformer sentiment mode or record a hardware preflight failure."""
    mode = str(config.model.get("mode", "lora"))
    if mode == "baseline":
        from transformer_lab.experiments.baseline import run_baseline

        return run_baseline(config)
    if mode not in {"head_only", "lora", "full"}:
        raise ValueError("sentiment mode must be baseline, head_only, lora, or full")
    if device.type != "cuda":
        return _unavailable(
            config,
            device,
            "GPT-2 sentiment runs require CUDA in this profile; the selected device is CPU",
        )
    run_seed = int(seed if seed is not None else config.evaluation.get("seed", 17))
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    revision = str(config.model.get("model_revision", REVISION))
    archive = Path(
        hf_hub_download(
            REPOSITORY,
            ARCHIVE,
            repo_type="dataset",
            revision=str(config.data.get("dataset_revision")),
        )
    )
    examples = load_phrasebank(archive, str(config.data.get("subset", "75Agree")))
    split_seed = int(config.data.get("split_seed", 17))
    splits = split_phrasebank(examples, seed=split_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model.get("base_model", "openai-community/gpt2")),
        revision=revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(config.data.get("max_length", 64))
    train_examples, validation_examples, evaluation_examples = _training_partitions(
        splits, validation_only=validation_only
    )
    encoded = [
        _encode(tokenizer, list(partition), max_length)
        for partition in (train_examples, validation_examples)
    ]
    if evaluation_examples is not None:
        encoded.append(_encode(tokenizer, list(evaluation_examples), max_length))
    train_dataset = TensorDataset(*encoded[0])
    validation_dataset = TensorDataset(*encoded[1])
    batch_size = int(config.optimization.get("batch_size", 1))
    train_generator = torch.Generator().manual_seed(run_seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    reference = AutoModelForCausalLM.from_pretrained(
        str(config.model.get("base_model", "openai-community/gpt2")),
        revision=revision,
    ).eval()
    model = SentimentClassifier(
        gpt2_small_config(), num_labels=int(config.model.get("num_labels", 3))
    )
    load_huggingface_gpt2_weights(model.backbone, reference.state_dict())
    del reference
    if mode == "head_only":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    elif mode == "lora":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        target_modules = tuple(
            config.model.get("target_modules", ["attention.c_attn", "attention.c_proj"])
        )
        inject_lora(
            model.backbone,
            LoRAConfig(
                rank=int(config.model.get("lora_rank", 4)),
                alpha=float(config.model.get("lora_alpha", 8)),
                dropout=float(config.model.get("lora_dropout", 0)),
                target_modules=target_modules,
            ),
        )
    model.to(device)
    optimisation = config.optimization
    output_directory = Path(config.output.get("directory", "artifacts/sentiment"))
    output_directory.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    result = _train_transformer(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=int(optimisation.get("epochs", 3)),
        learning_rate=float(optimisation.get("learning_rate", 5e-5)),
        weight_decay=float(optimisation.get("weight_decay", 0.01)),
        accumulation=int(optimisation.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(optimisation.get("max_grad_norm", 1.0)),
        amp=bool(optimisation.get("amp", True)),
        checkpoint_path=output_directory / "training.pt",
        best_model_path=output_directory / "best.pt",
        checkpoint_config=effective_config(config),
        resume=resume,
    )
    validation_targets, _, _, validation_logits = _predict(model, validation_loader, device)
    result["validation"]["loss"] = multiclass_log_loss(
        validation_targets,
        apply_temperature(validation_logits, 1.0),
        labels=list(range(len(LABELS))),
    )
    if validation_only:
        checkpoint = output_directory / "model.pt"
        _atomic_torch_save(
            _checkpoint_payload(
                model,
                mode=mode,
                revision=revision,
                config=config,
                max_length=max_length,
            ),
            checkpoint,
        )
        summary_path = validation_summary_path or output_directory / "validation-summary.json"
        return _write_validation_summary(
            summary_path,
            config=config,
            method=mode,
            validation=result["validation"],
            validation_candidates=result["validation_candidates"],
            train_seconds=float(result["train_seconds"]),
            data={
                "repository": REPOSITORY,
                "revision": config.data.get("dataset_revision"),
                "subset": config.data.get("subset", "75Agree"),
                "train_rows": len(train_examples),
                "validation_rows": len(validation_examples),
            },
            artifacts={"model": {"path": checkpoint.name, "sha256": file_sha256(checkpoint)}},
            candidate_id=candidate_id,
            identity_config=identity_config,
        )
    test_dataset = TensorDataset(*encoded[2])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    calibration_temperature = fit_temperature(validation_logits, validation_targets)
    targets, predictions, uncalibrated_probabilities, test_logits = _predict(
        model, test_loader, device
    )
    probabilities = apply_temperature(test_logits, calibration_temperature)
    uncalibrated_test = classification_metrics(
        targets,
        predictions,
        labels=list(range(len(LABELS))),
        probabilities=uncalibrated_probabilities,
        bootstrap_samples=int(config.evaluation.get("bootstrap_samples", 1_000)),
        seed=run_seed,
    )
    result["test"] = classification_metrics(
        targets,
        predictions,
        labels=list(range(len(LABELS))),
        probabilities=probabilities,
        bootstrap_samples=int(config.evaluation.get("bootstrap_samples", 1_000)),
        seed=run_seed,
    )
    counts = parameter_report(model)
    checkpoint = output_directory / "model.pt"
    checkpoint_payload = _checkpoint_payload(
        model,
        mode=mode,
        revision=revision,
        config=config,
        max_length=max_length,
        calibration_temperature=calibration_temperature,
    )
    _atomic_torch_save(checkpoint_payload, checkpoint)
    _atomic_torch_save(
        {
            "input_ids": encoded[2][0],
            "attention_mask": encoded[2][1],
            "targets": encoded[2][2],
            "predictions": torch.from_numpy(predictions),
            "uncalibrated_probabilities": torch.from_numpy(uncalibrated_probabilities),
            "probabilities": torch.from_numpy(probabilities),
            "labels": LABELS,
            "example_ids": evaluation_example_ids(splits.test),
        },
        output_directory / "evaluation.pt",
    )
    model_sha256 = file_sha256(checkpoint)
    evaluation_sha256 = file_sha256(output_directory / "evaluation.pt")
    provenance = source_provenance(config.source_path)
    artifacts = {
        "model": {"path": "model.pt", "sha256": model_sha256},
        "evaluation": {
            "path": "evaluation.pt",
            "sha256": evaluation_sha256,
        },
    }
    data_identity = {
        "split": split_summary(splits),
        "ordered_evaluation_rows": len(splits.test),
    }
    record = ExperimentRecord(
        run_id=str(config.experiment["name"]),
        status="completed",
        model={"id": config.model.get("base_model"), "revision": revision, "mode": mode},
        config=effective_config(config),
        data={
            "repository": REPOSITORY,
            "revision": config.data.get("dataset_revision"),
            "archive_member": ARCHIVE_MEMBER,
            "archive_sha256": _sha256(archive),
            "examples": len(examples),
            "unique_sentence_groups": len({example.group_id for example in examples}),
            "class_counts": dict(Counter(example.label for example in examples)),
            "split": split_summary(splits),
        },
        optimization=config.optimization,
        environment=environment_metadata(device),
        provenance=provenance,
        artifacts=artifacts,
        manifest=build_manifest(
            config,
            provenance=provenance,
            data_identity=data_identity,
            evaluation={
                "seed": run_seed,
                "bootstrap_samples": config.evaluation.get("bootstrap_samples", 1_000),
            },
            artifacts=artifacts,
        ),
        parameters={
            "total": counts.total,
            "trainable": counts.trainable,
            "trainable_share": counts.trainable / counts.total,
        },
        timing={
            "train_seconds": result["train_seconds"],
            "examples_per_second": (
                len(splits.train) * int(optimisation.get("epochs", 3)) / result["train_seconds"]
            ),
        },
        memory=peak_memory(device),
        metrics={
            "validation": result["validation"],
            "validation_candidates": result["validation_candidates"],
            "best_epoch": result["best_epoch"],
            "test": result["test"],
            "calibration": {
                "method": "temperature_scaling",
                "validation_temperature": calibration_temperature,
                "before": {
                    key: uncalibrated_test[key]
                    for key in ("brier_score", "expected_calibration_error")
                },
                "after": {
                    key: result["test"][key]
                    for key in ("brier_score", "expected_calibration_error")
                },
            },
        },
        schema_version=3,
    )
    result_path = Path(config.output["result"])
    record.write(result_path)
    return result_path


def _resolve_output_directory(config: ExperimentConfig, payload: dict[str, Any], mode: str) -> Path:
    """Resolve matrix artefacts when a base config names a seeded result file."""
    configured = Path(config.output.get("directory", "artifacts/sentiment"))
    result_name = Path(str(config.output.get("result", ""))).name
    match = re.search(r"(?:^|-)seed-(\d+)(?:\.[^.]+)?$", result_name)
    if match is None:
        return configured
    base_name = re.sub(r"-seed-\d+$", "", configured.name)
    candidate = configured.with_name(f"{base_name}-seed-{match.group(1)}")
    model_name, evaluation_name = (
        ("model.joblib", "evaluation.joblib")
        if mode == "baseline"
        else ("model.pt", "evaluation.pt")
    )
    model_path = candidate / model_name
    evaluation_path = candidate / evaluation_name
    if not model_path.exists() or not evaluation_path.exists():
        return configured
    expected_artifacts = payload.get("artifacts", {})
    expected_model = expected_artifacts.get("model", {})
    expected_evaluation = expected_artifacts.get("evaluation", {})
    if (
        isinstance(expected_model, dict)
        and expected_model.get("sha256")
        and file_sha256(model_path) != expected_model["sha256"]
    ):
        return configured
    if (
        isinstance(expected_evaluation, dict)
        and expected_evaluation.get("sha256")
        and file_sha256(evaluation_path) != expected_evaluation["sha256"]
    ):
        return configured
    return candidate


def _metrics_match(expected: Any, observed: Any) -> bool:
    """Compare JSON metrics while allowing harmless CPU/GPU float round-off."""
    if isinstance(expected, bool) or isinstance(observed, bool):
        return expected is observed
    if isinstance(expected, int | float) and isinstance(observed, int | float):
        return math.isclose(float(expected), float(observed), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(expected, dict) and isinstance(observed, dict):
        return expected.keys() == observed.keys() and all(
            _metrics_match(expected[key], observed[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            _metrics_match(left, right) for left, right in zip(expected, observed, strict=True)
        )
    return expected == observed


def evaluate_sentiment_run(config: ExperimentConfig, device: torch.device) -> Path:
    """Recompute test metrics from a saved model and frozen evaluation inputs."""
    config = resolve_run(config)
    result_path = Path(config.output["result"])
    if not result_path.exists():
        raise FileNotFoundError(f"sentiment result does not exist: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("sentiment run is not complete; run train-sentiment first")
    mode = str(config.model.get("mode", "lora"))
    output_dir = _resolve_output_directory(config, payload, mode)
    if mode == "baseline":
        evaluation_path = output_dir / "evaluation.joblib"
        model_path = output_dir / "model.joblib"
        if not evaluation_path.exists() or not model_path.exists():
            raise FileNotFoundError("saved baseline evaluation artifacts are missing")
        _, classifier = load(model_path)
        frozen = load(evaluation_path)
        predictions = classifier.predict(frozen["features"])
        probabilities = classifier.predict_proba(frozen["features"])
        measured = classification_metrics(
            np.asarray(frozen["targets"]),
            np.asarray(predictions),
            labels=list(frozen["labels"]),
            probabilities=np.asarray(probabilities),
            bootstrap_samples=int(config.evaluation.get("bootstrap_samples", 1_000)),
            seed=int(config.evaluation.get("seed", 17)),
        )
    else:
        evaluation_path = output_dir / "evaluation.pt"
        model_path = output_dir / "model.pt"
        if not evaluation_path.exists() or not model_path.exists():
            raise FileNotFoundError("saved transformer evaluation artifacts are missing")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        architecture = GPTConfig(**checkpoint["architecture"])
        model = SentimentClassifier(architecture, num_labels=int(checkpoint["num_labels"]))
        if checkpoint["mode"] == "lora":
            inject_lora(model.backbone, LoRAConfig(**checkpoint["lora"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        frozen = torch.load(evaluation_path, map_location="cpu", weights_only=True)
        loader = DataLoader(
            TensorDataset(frozen["input_ids"], frozen["attention_mask"], frozen["targets"]),
            batch_size=int(config.optimization.get("batch_size", 1)),
        )
        targets, predictions, _, logits = _predict(model, loader, device)
        probabilities = apply_temperature(
            logits, float(checkpoint.get("calibration_temperature", 1.0))
        )
        measured = classification_metrics(
            targets,
            predictions,
            labels=list(range(int(checkpoint["num_labels"]))),
            probabilities=probabilities,
            bootstrap_samples=int(config.evaluation.get("bootstrap_samples", 1_000)),
            seed=int(config.evaluation.get("seed", 17)),
        )
    if not _metrics_match(measured, payload.get("metrics", {}).get("test")):
        raise RuntimeError("recomputed test metrics do not match the recorded result")
    return result_path


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
