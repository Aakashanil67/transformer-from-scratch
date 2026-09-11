"""Application functions behind the ``transformer-lab`` command line tool."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from transformer_lab.config_io import ExperimentConfig, effective_config, load_experiment_config
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import environment_metadata, peak_memory
from transformer_lab.models.bigram import BigramLanguageModel
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
    if config.model.get("model_type", "bigram") != "bigram":
        raise ValueError("the current train-lm command supports model_type = 'bigram'")
    train_path = Path(config.data["train"])
    validation_path = Path(config.data["validation"])
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("prepared train and validation text files are required")
    device = resolve_device(device_name)
    run_seed = int(seed if seed is not None else config.generation.get("seed", 17))
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    model = BigramLanguageModel(int(config.model.get("vocab_size", 256))).to(device)
    train_ids = torch.tensor(list(train_path.read_bytes()), dtype=torch.long)
    validation_ids = torch.tensor(list(validation_path.read_bytes()), dtype=torch.long)
    optimisation = config.optimization
    checkpoint = Path(config.output.get("checkpoint", "artifacts/lm/model.pt"))
    stable_config = effective_config(config)
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
    )
    elapsed = history.elapsed_seconds
    record = ExperimentRecord(
        run_id=str(config.experiment["name"]),
        status="completed",
        model={"type": "bigram", "vocab_size": model.token_logits.num_embeddings},
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
        metrics={
            "training_loss": history.training_losses,
            "validation_loss": history.validation_losses,
            "validation_steps": history.validation_steps,
            "tokens_seen": history.tokens_seen,
            "checkpoint": stable_config["output"].get("checkpoint", str(checkpoint)),
        },
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
    model_type = payload.get("model_type", checkpoint_config.get("model", {}).get("model_type"))
    if model_type != "bigram":
        raise ValueError("sample currently supports bigram checkpoints")
    vocab_size = payload.get("vocab_size", checkpoint_config.get("model", {}).get("vocab_size"))
    model = BigramLanguageModel(int(vocab_size)).to(device)
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


def evaluate_sentiment(config_path: Path, *, device_name: str = "auto") -> Path:
    """Evaluate a completed sentiment run from its local checkpoint."""
    from transformer_lab.experiments.sentiment import evaluate_sentiment_run

    return evaluate_sentiment_run(load_experiment_config(config_path), resolve_device(device_name))
