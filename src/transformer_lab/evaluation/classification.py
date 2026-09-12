"""Classification metrics with a deterministic bootstrap interval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (examples, classes)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Choose a scalar temperature on validation negative log-likelihood."""
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("targets must align with two-dimensional logits")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("targets contain an out-of-range class")
    candidates = np.geomspace(0.05, 20.0, num=2_000)
    rows = np.arange(len(targets))
    losses = []
    for temperature in candidates:
        probabilities = apply_temperature(logits, float(temperature))
        losses.append(-np.log(np.clip(probabilities[rows, targets], 1e-12, 1.0)).mean())
    return float(candidates[int(np.argmin(losses))])


def paired_bootstrap_difference(
    targets: np.ndarray,
    challenger: np.ndarray,
    reference: np.ndarray,
    *,
    labels: Sequence[int | str],
    samples: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    """Compare aligned predictions by resampling the same rows for both methods."""
    if targets.shape != challenger.shape or targets.shape != reference.shape:
        raise ValueError("targets and both prediction arrays must have the same shape")
    if targets.ndim != 1 or not len(targets):
        raise ValueError("paired predictions must be non-empty one-dimensional arrays")
    if samples <= 0:
        raise ValueError("samples must be positive")
    label_list = list(labels)

    label_index = {label: index for index, label in enumerate(label_list)}
    target_ids = np.asarray([label_index[value] for value in targets], dtype=np.int64)

    def score(prediction: np.ndarray, rows: np.ndarray) -> float:
        predicted_ids = np.asarray([label_index[value] for value in prediction[rows]])
        matrix = np.bincount(
            target_ids[rows] * len(label_list) + predicted_ids,
            minlength=len(label_list) ** 2,
        ).reshape(len(label_list), len(label_list))
        true_positive = np.diag(matrix).astype(float)
        precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
        recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return float(f1.mean())

    all_rows = np.arange(len(targets))
    observed = score(challenger, all_rows) - score(reference, all_rows)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for index in range(samples):
        rows = rng.integers(0, len(targets), size=len(targets))
        differences[index] = score(challenger, rows) - score(reference, rows)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "macro_f1_difference": float(observed),
        "bootstrap_ci": {"lower": float(lower), "upper": float(upper)},
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "losses": int(np.sum(differences < 0)),
        "samples": samples,
    }


def summarise_seed_metrics(runs: Sequence[dict[str, float | int]]) -> dict[str, object]:
    if len(runs) < 2:
        raise ValueError("at least two seeds are required to estimate training variation")
    ordered = sorted(runs, key=lambda run: int(run["seed"]))
    summary: dict[str, object] = {"seeds": [int(run["seed"]) for run in ordered]}
    for metric in ("macro_f1", "accuracy"):
        values = np.asarray([float(run[metric]) for run in ordered])
        summary[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    labels: Sequence[int | str],
    probabilities: np.ndarray | None = None,
    bootstrap_samples: int = 1_000,
    seed: int = 17,
) -> dict[str, object]:
    """Return primary, per-class, and bootstrap macro-F1 metrics."""
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    label_list = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=label_list, zero_division=0
    )
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(targets), size=len(targets))
        bootstrap.append(
            f1_score(
                targets[indices],
                predictions[indices],
                labels=label_list,
                average="macro",
                zero_division=0,
            )
        )
    interval = np.quantile(bootstrap, [0.025, 0.975])
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_list,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(label_list)
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=label_list).tolist(),
        "macro_f1_bootstrap_ci": {"lower": float(interval[0]), "upper": float(interval[1])},
        "probabilities_recorded": probabilities is not None,
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(targets), len(label_list)):
            raise ValueError("probabilities must have shape (examples, labels)")
        if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("probability rows must be non-negative and sum to one")
        label_index = {label: index for index, label in enumerate(label_list)}
        target_indices = np.asarray([label_index[target] for target in targets])
        observed = np.eye(len(label_list))[target_indices]
        result["brier_score"] = float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))
        confidence = probabilities.max(axis=1)
        correct = predictions == targets
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            upper = lower + 0.1
            members = (confidence > lower) & (confidence <= upper)
            if np.any(members):
                ece += float(members.mean()) * abs(
                    float(correct[members].mean()) - float(confidence[members].mean())
                )
        result["expected_calibration_error"] = ece
    return result
