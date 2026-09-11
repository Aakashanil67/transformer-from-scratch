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
    return {
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
