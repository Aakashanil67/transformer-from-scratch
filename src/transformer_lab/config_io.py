"""Load the small TOML configuration files used by the experiment commands."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_SECTIONS = {"experiment", "model", "data", "optimization", "evaluation", "output", "generation"}
_KEYS = {
    "experiment": {"name", "kind", "description"},
    "model": {
        "model_type",
        "base_model",
        "model_revision",
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "num_labels",
        "mode",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "tolerance",
        "prompts",
        "batch_prompts",
        "random_token_lengths",
        "hidden_state_tolerance",
    },
    "data": {
        "raw_file",
        "train",
        "validation",
        "test",
        "archive",
        "subset",
        "dataset_revision",
        "max_length",
        "seed",
        "split_seed",
    },
    "optimization": {
        "steps",
        "epochs",
        "batch_size",
        "block_size",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "amp",
        "eval_interval",
        "eval_batches",
    },
    "evaluation": {"eval_interval", "eval_batches", "bootstrap_samples", "seed"},
    "output": {"directory", "checkpoint", "result", "overwrite"},
    "generation": {"prompt", "max_new_tokens", "temperature", "top_k", "do_sample", "seed"},
}
_PATH_KEYS = {
    "raw_file",
    "train",
    "validation",
    "test",
    "archive",
    "directory",
    "checkpoint",
    "result",
    "tokenizer",
}
_KINDS = {"lm", "gpt2-parity", "sentiment"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration plus the file from which it was loaded."""

    experiment: Mapping[str, Any]
    model: Mapping[str, Any]
    data: Mapping[str, Any]
    optimization: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    output: Mapping[str, Any]
    generation: Mapping[str, Any]
    source_path: Path

    def __post_init__(self) -> None:
        for name in _SECTIONS:
            object.__setattr__(self, name, _freeze(dict(getattr(self, name))))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _resolve_paths(section: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = dict(section)
    for key, value in resolved.items():
        if key in _PATH_KEYS and isinstance(value, str):
            path = Path(value)
            resolved[key] = str(path if path.is_absolute() else (root / path).resolve())
    return resolved


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment TOML file."""
    path = path.expanduser().resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"configuration file does not exist: {path}") from error
    unknown_sections = sorted(set(raw) - _SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown section '{unknown_sections[0]}'")
    if "experiment" not in raw:
        raise ValueError("missing required section 'experiment'")

    sections: dict[str, dict[str, Any]] = {}
    for section_name in _SECTIONS:
        values = raw.get(section_name, {})
        if not isinstance(values, dict):
            raise ValueError(f"section '{section_name}' must be a table")
        unknown_keys = sorted(set(values) - _KEYS[section_name])
        if unknown_keys:
            raise ValueError(f"unknown key '{section_name}.{unknown_keys[0]}'")
        sections[section_name] = _resolve_paths(values, path.parent)

    experiment = sections["experiment"]
    name = experiment.get("name")
    kind = experiment.get("kind")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment.name must be a non-empty string")
    if kind not in _KINDS:
        choices = ", ".join(sorted(_KINDS))
        raise ValueError(f"experiment.kind must be one of: {choices}")
    mode = sections["model"].get("mode")
    if kind == "sentiment" and mode not in {"baseline", "head_only", "lora", "full"}:
        raise ValueError("model.mode must be one of: baseline, head_only, lora, full")
    for section_name in ("optimization", "evaluation"):
        for key, value in sections[section_name].items():
            if key.endswith("steps") or key.endswith("batches") or key.endswith("interval"):
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{section_name}.{key} must be a positive integer")
    if (
        "learning_rate" in sections["optimization"]
        and sections["optimization"]["learning_rate"] <= 0
    ):
        raise ValueError("optimization.learning_rate must be positive")
    positive_fields = {
        "model": (
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "num_labels",
            "lora_rank",
            "lora_alpha",
            "tolerance",
        ),
        "data": ("max_length",),
        "optimization": (
            "steps",
            "epochs",
            "batch_size",
            "block_size",
            "learning_rate",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "eval_interval",
            "eval_batches",
        ),
        "evaluation": ("eval_interval", "eval_batches", "bootstrap_samples"),
        "generation": ("max_new_tokens", "temperature", "top_k"),
    }
    for section_name, fields in positive_fields.items():
        for key in fields:
            if key not in sections[section_name]:
                continue
            value = sections[section_name][key]
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{section_name}.{key} must be positive")

    return ExperimentConfig(source_path=path, **sections)


def effective_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return a stable, JSON-safe representation for experiment records."""
    sections = {
        name: dict(getattr(config, name))
        for name in (
            "experiment",
            "model",
            "data",
            "optimization",
            "evaluation",
            "output",
            "generation",
        )
    }
    base = config.source_path.parent.resolve()
    for _section_name, values in sections.items():
        for key, value in values.items():
            if key not in _PATH_KEYS or not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_absolute():
                continue
            try:
                values[key] = path.resolve().relative_to(base).as_posix()
            except ValueError:
                values[key] = Path(os.path.relpath(path.resolve(), base)).as_posix()
    payload = {
        "experiment": sections["experiment"],
        "model": sections["model"],
        "data": sections["data"],
        "optimization": sections["optimization"],
        "evaluation": sections["evaluation"],
        "output": sections["output"],
        "generation": sections["generation"],
    }
    return json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
