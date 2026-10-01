"""Deterministic, checked rendering helpers for tracked experiment summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_METHOD_NAMES = {
    "tfidf": "TF-IDF + balanced logistic regression",
    "head_only": "GPT-2 head-only",
    "lora": "GPT-2 LoRA",
    "full": "GPT-2 full fine-tuning",
    "bigram": "Byte bigram",
    "transformer": "Scratch transformer",
}


def _validate_finite(value: Any, path: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise ValueError(f"result value at {path} must be finite")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, f"{path}[{index}]")


def validate_result_payload(payload: Mapping[str, Any]) -> None:
    """Validate finite JSON and the aggregate method summary contract."""
    if not isinstance(payload, Mapping):
        raise ValueError("result payload must be an object")
    _validate_finite(payload, "result")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("result payload has no metrics")
    methods = metrics.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("result payload has no method summaries")
    for method, summary in methods.items():
        if not isinstance(summary, Mapping):
            raise ValueError(f"method {method} summary must be an object")
        seeds = summary.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not isinstance(seed, int) for seed in seeds)
        ):
            raise ValueError(f"method {method} must declare integer seeds")
        for metric_name in ("macro_f1", "accuracy"):
            metric = summary.get(metric_name)
            if not isinstance(metric, Mapping):
                raise ValueError(f"method {method} has no {metric_name} summary")
            mean = metric.get("mean")
            standard_deviation = metric.get("standard_deviation")
            if isinstance(mean, bool) or not isinstance(mean, int | float):
                raise ValueError(f"method {method} {metric_name}.mean must be numeric")
            if len(seeds) == 1 and standard_deviation is not None:
                raise ValueError(f"single-seed method {method} must not report seed SD")
            if len(seeds) > 1 and (
                isinstance(standard_deviation, bool)
                or not isinstance(standard_deviation, int | float)
            ):
                raise ValueError(f"multi-seed method {method} must report seed SD")


def format_gib(byte_count: int | float | None) -> str:
    """Format bytes as binary GiB, never decimal GB."""
    if byte_count is None:
        return "n/a"
    if isinstance(byte_count, bool) or not isinstance(byte_count, int | float):
        raise ValueError("memory must be numeric bytes")
    if byte_count < 0 or not math.isfinite(float(byte_count)):
        raise ValueError("memory must be finite and non-negative")
    return f"{byte_count / (1024**3):.3f} GiB"


def _mean_sd(metric: Mapping[str, Any]) -> str:
    mean = float(metric["mean"])
    standard_deviation = metric.get("standard_deviation")
    if standard_deviation is None:
        return f"{mean:.4f} (single seed)"
    return f"{mean:.4f} ± {float(standard_deviation):.4f}"


def render_comparison_markdown(payload: Mapping[str, Any], *, title: str = "Results") -> str:
    """Render the aggregate comparison table with a stable five-column schema."""
    validate_result_payload(payload)
    methods = payload["metrics"]["methods"]
    lines = [
        f"## {title}",
        "",
        "| Method | Test macro-F1 (mean ± SD) | Accuracy (mean ± SD) | "
        "Peak allocated memory (GiB) | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for method, summary in methods.items():
        name = _METHOD_NAMES.get(str(method), str(method))
        memory = summary.get("peak_allocated_bytes", summary.get("peak_allocated"))
        lines.append(
            f"| {name} | {_mean_sd(summary['macro_f1'])} | {_mean_sd(summary['accuracy'])} | "
            f"{format_gib(memory)} | {payload.get('status', 'unknown')} |"
        )
    return "\n".join(lines) + "\n"


def render_class_metrics_markdown(record: Mapping[str, Any]) -> str:
    """Render per-class support, recall and F1 from one run record."""
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("record has no metrics")
    test = metrics.get("test")
    if not isinstance(test, Mapping):
        raise ValueError("record has no test metrics")
    classes = test.get("per_class")
    if not isinstance(classes, Mapping):
        raise ValueError("record has no per-class metrics")
    lines = [
        "| Class | Support | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, values in classes.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"class {label} metrics must be an object")
        lines.append(
            f"| {label} | {int(values['support'])} | {float(values['recall']):.4f} | "
            f"{float(values['f1']):.4f} |"
        )
    return "\n".join(lines) + "\n"
