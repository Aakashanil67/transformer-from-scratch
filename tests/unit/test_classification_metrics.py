import numpy as np
import pytest

from transformer_lab.evaluation.classification import (
    apply_temperature,
    classification_metrics,
    fit_temperature,
    paired_bootstrap_difference,
    summarise_seed_metrics,
)


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


def test_probability_metrics_report_brier_score_and_calibration_error() -> None:
    metrics = classification_metrics(
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        labels=[0, 1, 2],
        probabilities=np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]], dtype=float),
        bootstrap_samples=5,
    )

    assert metrics["brier_score"] == pytest.approx(0.11333333333333336)
    assert metrics["expected_calibration_error"] == pytest.approx(0.2666666666666667)


def test_probability_metrics_reject_non_finite_or_wrong_class_probabilities() -> None:
    with pytest.raises(ValueError, match="finite"):
        classification_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            labels=[0, 1],
            probabilities=np.array([[np.nan, 0.0], [0.0, 1.0]]),
            bootstrap_samples=2,
        )

    with pytest.raises(ValueError, match="predicted classes"):
        classification_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            labels=[0, 1],
            probabilities=np.array([[0.1, 0.9], [0.0, 1.0]]),
            bootstrap_samples=2,
        )


def test_calibration_bins_include_the_exact_point_nine_boundary() -> None:
    metrics = classification_metrics(
        np.array([0, 1]),
        np.array([0, 0]),
        labels=[0, 1],
        probabilities=np.array([[0.9, 0.1], [0.9, 0.1]]),
        bootstrap_samples=2,
    )

    assert metrics["expected_calibration_error"] == pytest.approx(0.4)


def test_temperature_fit_reduces_validation_negative_log_likelihood() -> None:
    logits = np.array([[8.0, 0.0], [0.0, 8.0], [5.0, 0.0], [5.0, 0.0]])
    targets = np.array([0, 1, 0, 1])

    temperature = fit_temperature(logits, targets)
    calibrated = apply_temperature(logits, temperature)
    uncalibrated = apply_temperature(logits, 1.0)

    def nll(values: np.ndarray) -> float:
        return float(-np.log(values[np.arange(len(targets)), targets]).mean())

    assert temperature > 1.0
    assert nll(calibrated) < nll(uncalibrated)


def test_paired_bootstrap_uses_aligned_examples() -> None:
    targets = np.array([0, 0, 1, 1, 2, 2])
    reference = np.array([0, 0, 1, 1, 2, 0])
    challenger = np.array([0, 0, 1, 1, 2, 2])

    result = paired_bootstrap_difference(
        targets, challenger, reference, labels=[0, 1, 2], samples=100, seed=9
    )

    assert result["macro_f1_difference"] > 0
    assert result["wins"] + result["ties"] + result["losses"] == 100


def test_paired_bootstrap_can_resample_duplicate_groups_as_clusters() -> None:
    targets = np.array([0, 0, 1, 1])
    reference = np.array([0, 1, 1, 1])
    challenger = np.array([0, 0, 1, 0])

    result = paired_bootstrap_difference(
        targets,
        challenger,
        reference,
        groups=np.array(["a", "a", "b", "b"]),
        labels=[0, 1],
        samples=20,
        seed=9,
    )

    assert result["resampling_unit"] == "duplicate_group"
    assert result["group_count"] == 2


def test_seed_summary_keeps_training_variation_separate() -> None:
    summary = summarise_seed_metrics(
        [
            {"seed": 17, "macro_f1": 0.7, "accuracy": 0.8},
            {"seed": 23, "macro_f1": 0.8, "accuracy": 0.9},
            {"seed": 41, "macro_f1": 0.9, "accuracy": 1.0},
        ]
    )

    assert summary["seeds"] == [17, 23, 41]
    assert summary["macro_f1"]["mean"] == pytest.approx(0.8)
    assert summary["macro_f1"]["standard_deviation"] == pytest.approx(0.1)
