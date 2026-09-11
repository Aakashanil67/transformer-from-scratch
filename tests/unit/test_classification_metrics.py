import numpy as np

from transformer_lab.evaluation.classification import classification_metrics


def test_classification_metrics_reports_macro_f1_and_confusion_matrix() -> None:
    metrics = classification_metrics(
        np.array([0, 1, 2, 2]),
        np.array([0, 1, 1, 2]),
        labels=[0, 1, 2],
        bootstrap_samples=20,
        seed=7,
    )

    assert metrics["accuracy"] == 0.75
    assert 0 <= metrics["macro_f1"] <= 1
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert set(metrics["macro_f1_bootstrap_ci"]) == {"lower", "upper"}
