import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump

from transformer_lab.experiments.comparison import RunOutput, _predictions, build_comparison


def _result(path: Path, *, seed: int, macro_f1: float, accuracy: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "config": {"evaluation": {"seed": seed}, "data": {"split_seed": 17}},
                "data": {"split": {"fingerprint": "fixed-split"}},
                "metrics": {"test": {"macro_f1": macro_f1, "accuracy": accuracy}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_comparison_summarises_seeds_and_runs_paired_bootstrap(tmp_path: Path) -> None:
    targets = np.array([0, 0, 1, 1, 2, 2])
    baseline_predictions = np.array([0, 0, 1, 1, 2, 0])
    baseline_eval = tmp_path / "baseline.joblib"
    dump({"targets": targets, "predictions": baseline_predictions}, baseline_eval)
    runs = [
        RunOutput(
            "tfidf",
            17,
            _result(tmp_path / "baseline.json", seed=17, macro_f1=0.82, accuracy=0.83),
            baseline_eval,
        )
    ]
    for seed, predictions, macro_f1 in (
        (17, np.array([0, 0, 1, 1, 2, 2]), 0.90),
        (23, np.array([0, 0, 1, 2, 2, 2]), 0.84),
    ):
        evaluation = tmp_path / f"lora-{seed}.pt"
        torch.save(
            {"targets": torch.from_numpy(targets), "predictions": torch.from_numpy(predictions)},
            evaluation,
        )
        runs.append(
            RunOutput(
                "lora",
                seed,
                _result(
                    tmp_path / f"lora-{seed}.json",
                    seed=seed,
                    macro_f1=macro_f1,
                    accuracy=macro_f1,
                ),
                evaluation,
            )
        )

    destination = build_comparison(runs, tmp_path / "comparison.json", bootstrap_samples=100)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["data"]["split_fingerprint"] == "fixed-split"
    assert payload["metrics"]["methods"]["lora"]["seeds"] == [17, 23]
    assert len(payload["metrics"]["paired_comparisons"]["lora_minus_tfidf"]) == 2


def test_comparison_rejects_mixed_test_splits(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation.pt"
    torch.save({"targets": torch.tensor([0]), "predictions": torch.tensor([0])}, evaluation)
    first = _result(tmp_path / "one.json", seed=17, macro_f1=1.0, accuracy=1.0)
    second = _result(tmp_path / "two.json", seed=23, macro_f1=1.0, accuracy=1.0)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["data"]["split"]["fingerprint"] = "different"
    second.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_comparison(
            [RunOutput("lora", 17, first, evaluation), RunOutput("full", 23, second, evaluation)],
            tmp_path / "comparison.json",
        )
    except ValueError as error:
        assert "split" in str(error)
    else:
        raise AssertionError("mixed test splits were compared")


def test_comparison_accepts_id_only_evaluation_artifacts(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation.joblib"
    dump({"target_ids": np.array([0, 1]), "prediction_ids": np.array([1, 1])}, evaluation)

    targets, predictions = _predictions(evaluation)

    assert targets.tolist() == [0, 1]
    assert predictions.tolist() == [1, 1]
