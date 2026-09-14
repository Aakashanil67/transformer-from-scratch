"""Application functions behind the ``transformer-lab`` command line tool."""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from transformer_lab.config import GPTConfig
from transformer_lab.config_io import ExperimentConfig, effective_config, load_experiment_config
from transformer_lab.experiments.manifests import build_manifest, config_digest, verify_run
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import (
    environment_metadata,
    file_sha256,
    peak_memory,
    source_provenance,
)
from transformer_lab.models.bigram import BigramLanguageModel
from transformer_lab.models.transformer import DecoderOnlyTransformer
from transformer_lab.training import TrainingConfig, train_language_model


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device request without silently falling back from CUDA."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this PyTorch environment"
            )
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _result_path(config: ExperimentConfig) -> Path:
    result = config.output.get("result")
    if not isinstance(result, str):
        raise ValueError("output.result must be set")
    return Path(result)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def train_lm(
    config_path: Path,
    *,
    device_name: str = "auto",
    seed: int | None = None,
    resume: bool = False,
) -> Path:
    """Train the configured language model and save its checkpoint and result."""
    config = load_experiment_config(config_path)
    if config.experiment["kind"] != "lm":
        raise ValueError("train-lm requires a configuration with experiment.kind = 'lm'")
    model_type = str(config.model.get("model_type", "bigram"))
    if model_type not in {"bigram", "transformer"}:
        raise ValueError("train-lm model_type must be bigram or transformer")
    train_path = Path(config.data["train"])
    validation_path = Path(config.data["validation"])
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("prepared train and validation text files are required")
    device = resolve_device(device_name)
    run_seed = int(seed if seed is not None else config.generation.get("seed", 17))
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    optimisation = config.optimization
    if model_type == "bigram":
        model = BigramLanguageModel(int(config.model.get("vocab_size", 256))).to(device)
        model_metadata: dict[str, Any] = {
            "model_type": "bigram",
            "vocab_size": model.token_logits.num_embeddings,
        }
    else:
        architecture = GPTConfig(
            vocab_size=int(config.model.get("vocab_size", 256)),
            block_size=int(config.model.get("block_size", optimisation.get("block_size", 8))),
            n_layer=int(config.model["n_layer"]),
            n_head=int(config.model["n_head"]),
            n_embd=int(config.model["n_embd"]),
            dropout=float(config.model.get("dropout", 0.0)),
            bias=bool(config.model.get("bias", True)),
        )
        model = DecoderOnlyTransformer(architecture).to(device)
        model_metadata = {"model_type": "transformer", "architecture": architecture.__dict__}
    train_ids = torch.tensor(list(train_path.read_bytes()), dtype=torch.long)
    validation_ids = torch.tensor(list(validation_path.read_bytes()), dtype=torch.long)
    checkpoint = Path(config.output.get("checkpoint", "artifacts/lm/model.pt"))
    stable_config = effective_config(config)
    stable_config["generation"]["seed"] = run_seed
    manifest_config = replace(config, generation={**dict(config.generation), "seed": run_seed})
    history = train_language_model(
        model,
        train_token_ids=train_ids,
        validation_token_ids=validation_ids,
        config=TrainingConfig(
            steps=int(optimisation.get("steps", 100)),
            batch_size=int(optimisation.get("batch_size", 32)),
            block_size=int(optimisation.get("block_size", 8)),
            learning_rate=float(optimisation.get("learning_rate", 0.1)),
            eval_interval=int(optimisation.get("eval_interval", 25)),
            eval_batches=int(optimisation.get("eval_batches", 4)),
            seed=run_seed,
            weight_decay=float(optimisation.get("weight_decay", 0.01)),
            warmup_steps=int(optimisation.get("warmup_steps", 0)),
            gradient_accumulation_steps=int(optimisation.get("gradient_accumulation_steps", 1)),
            max_grad_norm=float(optimisation.get("max_grad_norm", 1.0)),
            amp=bool(optimisation.get("amp", False)),
        ),
        checkpoint_path=checkpoint,
        resume=resume,
        checkpoint_config=stable_config,
        checkpoint_metadata=model_metadata,
    )
    elapsed = history.elapsed_seconds
    record = ExperimentRecord(
        run_id=str(config.experiment["name"]),
        status="completed",
        model={"type": model_type, **model_metadata},
        config=stable_config,
        data={"seed": run_seed},
        optimization=dict(config.optimization),
        environment=environment_metadata(device),
        parameters={
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        timing={
            "train_seconds": elapsed,
            "tokens_per_second": history.tokens_seen / elapsed if elapsed else 0.0,
        },
        memory=peak_memory(device),
        artifacts={
            "model": {"path": checkpoint.name, "sha256": file_sha256(checkpoint)},
        },
        manifest=build_manifest(
            manifest_config,
            provenance=source_provenance(config_path),
            data_identity={
                "train_sha256": file_sha256(train_path),
                "validation_sha256": file_sha256(validation_path),
            },
            evaluation={
                "seed": run_seed,
                "eval_interval": int(optimisation.get("eval_interval", 25)),
                "eval_batches": int(optimisation.get("eval_batches", 4)),
            },
            artifacts={"model": {"path": checkpoint.name, "sha256": file_sha256(checkpoint)}},
        ),
        metrics={
            "training_loss": history.training_losses,
            "validation_loss": history.validation_losses,
            "validation_steps": history.validation_steps,
            "tokens_seen": history.tokens_seen,
            "optimizer_steps": history.optimizer_steps,
            "checkpoint": stable_config["output"].get("checkpoint", str(checkpoint)),
        },
        schema_version=3,
    )
    return record.write(_result_path(config))


def sample(config_path: Path, *, device_name: str = "auto", seed: int | None = None) -> str:
    """Load a local language-model checkpoint and return generated text."""
    config = load_experiment_config(config_path)
    checkpoint = Path(config.output.get("checkpoint", ""))
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_config = payload.get("config", {})
    model_type = payload.get(
        "model_type", checkpoint_config.get("model", {}).get("model_type", "bigram")
    )
    if model_type == "bigram":
        vocab_size = payload.get("vocab_size", checkpoint_config.get("model", {}).get("vocab_size"))
        if vocab_size is None:
            raise ValueError("bigram checkpoint has no vocabulary size")
        model = BigramLanguageModel(int(vocab_size)).to(device)
    elif model_type == "transformer":
        architecture_payload = payload.get("architecture")
        if not isinstance(architecture_payload, dict):
            raise ValueError("transformer checkpoint has no architecture")
        model = DecoderOnlyTransformer(GPTConfig(**architecture_payload)).to(device)
    else:
        raise ValueError(f"unsupported language-model checkpoint type: {model_type}")
    model.load_state_dict(payload.get("model", payload.get("state_dict")))
    model.eval()
    run_seed = int(seed if seed is not None else config.generation.get("seed", 17))
    prompt = str(config.generation.get("prompt", ""))
    token_ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    generated = model.generate(
        token_ids,
        max_new_tokens=int(config.generation.get("max_new_tokens", 80)),
        temperature=float(config.generation.get("temperature", 1.0)),
        top_k=(
            int(config.generation["top_k"]) if config.generation.get("top_k") is not None else None
        ),
        do_sample=bool(config.generation.get("do_sample", True)),
        generator=torch.Generator(device=device).manual_seed(run_seed),
    )
    return bytes(generated[0].tolist()).decode("utf-8", errors="replace")


def verify_gpt2(config_path: Path, *, device_name: str = "cpu") -> Path:
    """Run the pinned GPT-2 parity comparison."""
    from transformer_lab.experiments.parity import run

    config = load_experiment_config(config_path)
    return run(config, device=resolve_device(device_name))


def train_sentiment(
    config_path: Path,
    *,
    device_name: str = "auto",
    seed: int | None = None,
    resume: bool = False,
) -> Path:
    """Run the configured sentiment experiment."""
    from transformer_lab.experiments.sentiment import run_sentiment_experiment

    return run_sentiment_experiment(
        load_experiment_config(config_path),
        device=resolve_device(device_name),
        seed=seed,
        resume=resume,
    )


def train_sentiment_candidate(
    config_path: Path,
    *,
    summary_path: Path,
    device_name: str = "cuda",
    seed: int | None = None,
    resume: bool = False,
) -> Path:
    """Train one validation-only sentiment candidate without test evaluation."""
    from transformer_lab.experiments.sentiment import run_sentiment_candidate

    return run_sentiment_candidate(
        load_experiment_config(config_path),
        device=resolve_device(device_name),
        summary_path=summary_path,
        seed=seed,
        resume=resume,
    )


def evaluate_sentiment(config_path: Path, *, device_name: str = "auto") -> Path:
    """Evaluate a completed sentiment run from its local checkpoint."""
    from transformer_lab.experiments.sentiment import evaluate_sentiment_run

    return evaluate_sentiment_run(load_experiment_config(config_path), resolve_device(device_name))


def run_sentiment_matrix(
    config_paths: list[Path],
    *,
    seeds: list[int],
    device_name: str,
    output: Path,
    resume: bool = False,
    config_overrides: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Run a fixed-split seed matrix and write its paired comparison."""
    from transformer_lab.experiments.baseline import run_baseline
    from transformer_lab.experiments.comparison import RunOutput, build_comparison
    from transformer_lab.experiments.sentiment import run_sentiment_experiment, sentiment_seeds

    device = resolve_device(device_name)
    runs = []
    for path in config_paths:
        config = load_experiment_config(path)
        overrides = (config_overrides or {}).get(str(path.resolve()), {})
        if overrides:
            config = replace(
                config,
                model={**dict(config.model), **dict(overrides.get("model", {}))},
                data={**dict(config.data), **dict(overrides.get("data", {}))},
                optimization={
                    **dict(config.optimization),
                    **dict(overrides.get("optimization", {})),
                },
                evaluation={
                    **dict(config.evaluation),
                    **dict(overrides.get("evaluation", {})),
                },
                output={
                    **dict(config.output),
                    **dict(overrides.get("output", {})),
                },
            )
        mode = str(config.model.get("mode"))
        if mode == "baseline":
            baseline_config = config
            if resume and Path(baseline_config.output["result"]).exists():
                verify_run(
                    Path(baseline_config.output["result"]),
                    artifact_root=Path(baseline_config.output["directory"]),
                    expected_config=baseline_config,
                )
                result = Path(baseline_config.output["result"])
                runs.append(
                    RunOutput(
                        "tfidf",
                        int(config.evaluation.get("seed", 17)),
                        result,
                        Path(baseline_config.output["directory"]) / "evaluation.joblib",
                    )
                )
                continue
            result = run_baseline(config)
            runs.append(
                RunOutput(
                    "tfidf",
                    int(config.evaluation.get("seed", 17)),
                    result,
                    Path(config.output["directory"]) / "evaluation.joblib",
                )
            )
            continue
        for seeded in sentiment_seeds(config, seeds):
            result = Path(seeded.output["result"])
            output_dir = Path(seeded.output["directory"])
            evaluation = output_dir / "evaluation.pt"
            completed = False
            if resume and result.exists():
                verify_run(result, artifact_root=output_dir, expected_config=seeded)
                completed = True
            if not completed:
                result = run_sentiment_experiment(
                    seeded,
                    device=device,
                    seed=int(seeded.evaluation["seed"]),
                    resume=resume and (output_dir / "training.pt").exists(),
                )
            runs.append(
                RunOutput(
                    mode,
                    int(seeded.evaluation["seed"]),
                    result,
                    evaluation,
                )
            )
    return build_comparison(runs, output.resolve())


def select_sentiment(protocol_path: Path, *, output: Path | None = None) -> Path:
    """Freeze one validation-selected candidate per method without test access."""
    from transformer_lab.experiments.selection import (
        load_selection_protocol,
        write_selection_manifest,
    )

    protocol, candidates = load_selection_protocol(protocol_path.resolve())
    destination = output or protocol_path.with_name("selection.json")
    manifest_root = destination.resolve().parent
    for candidate in candidates:
        for key in ("config", "summary_path"):
            value = candidate.get(key)
            if not isinstance(value, str):
                continue
            source_path = Path(value)
            if not source_path.is_absolute():
                source_path = (protocol_path.resolve().parent / source_path).resolve()
            candidate[key] = Path(os.path.relpath(source_path, manifest_root)).as_posix()
    return write_selection_manifest(destination.resolve(), protocol=protocol, candidates=candidates)


def evaluate_matrix(
    selection_path: Path,
    *,
    seeds: list[int],
    device_name: str,
    output: Path,
    resume: bool = False,
) -> Path:
    """Evaluate only the configurations frozen by a validated selection manifest."""
    from transformer_lab.experiments.selection import load_selection_manifest

    selection = load_selection_manifest(selection_path.resolve())
    config_paths: list[Path] = []
    config_overrides: dict[str, dict[str, Any]] = {}
    matrix_root = output.resolve().parent / "runs"
    for method, chosen in selection["selection"].items():
        if chosen.get("status") != "completed":
            raise RuntimeError(f"selection has no completed candidate for method {method}")
        config = chosen.get("config")
        if not isinstance(config, str) or not config:
            raise ValueError(f"selection candidate for {method} has no config path")
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = (selection_path.resolve().parent / config_path).resolve()
        expected_digest = chosen.get("config_digest")
        if expected_digest is not None:
            if not isinstance(expected_digest, str):
                raise ValueError(f"selection candidate for {method} has invalid config digest")
            selected_config = load_experiment_config(config_path)
            overrides = chosen.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"selection candidate for {method} has invalid overrides")
            selected_config = replace(
                selected_config,
                model={**dict(selected_config.model), **dict(overrides.get("model", {}))},
                data={**dict(selected_config.data), **dict(overrides.get("data", {}))},
                optimization={
                    **dict(selected_config.optimization),
                    **dict(overrides.get("optimization", {})),
                },
                evaluation={
                    **dict(selected_config.evaluation),
                    **dict(overrides.get("evaluation", {})),
                },
            )
            if config_digest(selected_config) != expected_digest:
                raise ValueError(f"selection candidate for {method} has a stale configuration")
        config_paths.append(config_path)
        overrides = chosen.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"selection candidate for {method} has invalid overrides")
        config_overrides[str(config_path)] = {
            **overrides,
            "output": {
                "directory": str(matrix_root / method),
                "result": str(matrix_root / f"{method}.json"),
            },
        }
    if not config_paths:
        raise ValueError("selection manifest contains no configurations")
    return run_sentiment_matrix(
        config_paths,
        seeds=seeds,
        device_name=device_name,
        output=output,
        resume=resume,
        config_overrides=config_overrides,
    )
