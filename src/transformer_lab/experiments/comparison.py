"""Aggregate repeated sentiment runs without treating test rows as training seeds."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from joblib import load

from transformer_lab.evaluation.classification import (
    classification_metrics,
    paired_bootstrap_difference,
    summarise_seed_metrics,
)
from transformer_lab.experiments.manifests import verify_run
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import source_provenance


@dataclass(frozen=True)
class RunOutput:
    method: str
    seed: int
    result: Path
    evaluation: Path


@dataclass(frozen=True)
class EvaluationEvidence:
    targets: np.ndarray
    predictions: np.ndarray
    example_ids: np.ndarray | None = None
    labels: tuple[object, ...] | None = None
    probabilities: np.ndarray | None = None
    uncalibrated_probabilities: np.ndarray | None = None


def _as_array(value: object, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"evaluation artifact has no {name}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_evaluation(path: Path) -> EvaluationEvidence:
    if path.suffix == ".joblib":
        payload = load(path)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation artifact is not a mapping: {path.name}")
    targets = _as_array(payload.get("target_ids", payload.get("targets")), "targets")
    predictions = _as_array(
        payload.get("prediction_ids", payload.get("predictions")), "predictions"
    )
    if targets.ndim != 1 or predictions.ndim != 1 or targets.shape != predictions.shape:
        raise ValueError(
            "evaluation targets and predictions must be aligned one-dimensional arrays"
        )
    raw_ids = payload.get("example_ids")
    example_ids = None if raw_ids is None else np.asarray(raw_ids, dtype=str)
    if example_ids is not None and example_ids.shape != targets.shape:
        raise ValueError("evaluation example IDs must align with targets")
    raw_labels = payload.get("labels")
    labels = None if raw_labels is None else tuple(raw_labels)
    probabilities = payload.get("probabilities")
    if probabilities is not None:
        probabilities = _as_array(probabilities, "probabilities").astype(float)
        _validate_probabilities(probabilities, predictions, labels)
    uncalibrated = payload.get("uncalibrated_probabilities")
    if uncalibrated is not None:
        uncalibrated = _as_array(uncalibrated, "uncalibrated_probabilities").astype(float)
        _validate_probabilities(uncalibrated, predictions, labels)
    return EvaluationEvidence(
        targets,
        predictions,
        example_ids,
        labels,
        probabilities,
        uncalibrated,
    )


def _predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    evidence = _load_evaluation(path)
    return evidence.targets, evidence.predictions


def _metric_labels(evidence: EvaluationEvidence) -> list[object]:
    values = np.concatenate((evidence.targets, evidence.predictions))
    if evidence.labels is not None and np.issubdtype(values.dtype, np.integer):
        return list(range(len(evidence.labels)))
    return np.unique(values).tolist()


def _validate_probabilities(
    probabilities: np.ndarray, predictions: np.ndarray, labels: tuple[object, ...] | None
) -> None:
    if probabilities.ndim != 2 or probabilities.shape[0] != len(predictions):
        raise ValueError("evaluation probabilities must align with predictions")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("evaluation probabilities must be finite")
    if np.any(probabilities < -1e-8) or np.any(probabilities > 1 + 1e-8):
        raise ValueError("evaluation probabilities must be bounded between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("evaluation probability rows must sum to one")
    predicted_ids = probabilities.argmax(axis=1)
    if labels is not None and not np.issubdtype(predictions.dtype, np.integer):
        label_index = {label: index for index, label in enumerate(labels)}
        try:
            expected_ids = np.asarray([label_index[value] for value in predictions])
        except KeyError as error:
            raise ValueError("evaluation predictions contain an unknown class label") from error
    else:
        expected_ids = predictions.astype(int)
    if not np.array_equal(predicted_ids, expected_ids):
        raise ValueError("evaluation probabilities disagree with predicted classes")


def _headline_metrics(evidence: EvaluationEvidence) -> dict[str, float]:
    labels = _metric_labels(evidence)
    metrics = classification_metrics(
        evidence.targets,
        evidence.predictions,
        labels=labels,
        probabilities=evidence.probabilities,
        bootstrap_samples=1,
    )
    return {"macro_f1": float(metrics["macro_f1"]), "accuracy": float(metrics["accuracy"])}


def _assert_record_metrics(record: dict[str, object], observed: dict[str, float]) -> None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("test"), dict):
        raise ValueError("comparison input has no test metrics")
    expected = metrics["test"]
    for key in ("macro_f1", "accuracy"):
        value = expected.get(key)
        if not isinstance(value, int | float) or not np.isfinite(value):
            raise ValueError(f"comparison input has no finite test {key}")
        if not np.isclose(float(value), observed[key], rtol=1e-9, atol=1e-9):
            raise ValueError(f"recorded test metrics do not match saved predictions ({key})")


def _method_summary(rows: list[dict[str, float | int]]) -> dict[str, object]:
    if len(rows) > 1:
        return summarise_seed_metrics(rows)
    row = rows[0]
    return {
        "seeds": [int(row["seed"])],
        "macro_f1": {"mean": float(row["macro_f1"]), "standard_deviation": None},
        "accuracy": {"mean": float(row["accuracy"]), "standard_deviation": None},
    }


def build_comparison(
    runs: list[RunOutput],
    destination: Path,
    *,
    bootstrap_samples: int = 2_000,
) -> Path:
    if not runs:
        raise ValueError("comparison requires at least one completed run")
    split_fingerprints = set()
    metrics_by_method: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    predictions: dict[tuple[str, int], EvaluationEvidence] = {}
    schema_by_key: dict[tuple[str, int], int] = {}
    seen_keys: set[tuple[str, int]] = set()
    for run in runs:
        record = json.loads(run.result.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise ValueError(f"comparison input is not completed: {run.result.name}")
        split = record.get("data", {}).get("split", {})
        fingerprint = split.get("sha256", split.get("fingerprint"))
        if not fingerprint:
            raise ValueError(f"comparison input has no split fingerprint: {run.result.name}")
        split_fingerprints.add(str(fingerprint))
        key = (run.method, int(run.seed))
        if key in seen_keys:
            raise ValueError(f"comparison input duplicates method and seed: {key}")
        seen_keys.add(key)
        if record.get("schema_version", 2) == 3:
            verify_run(run.result, artifact_root=run.evaluation.parent)
        evidence = _load_evaluation(run.evaluation)
        if evidence.example_ids is not None and len(np.unique(evidence.example_ids)) != len(
            evidence.example_ids
        ):
            raise ValueError("evaluation example IDs must be unique")
        observed = _headline_metrics(evidence)
        _assert_record_metrics(record, observed)
        metrics_by_method[run.method].append({"seed": run.seed, **observed})
        predictions[key] = evidence
        schema_by_key[key] = int(record.get("schema_version", 2))
    if len(split_fingerprints) != 1:
        raise ValueError("all comparison inputs must use the same test split")

    paired: dict[str, list[dict[str, object]]] = defaultdict(list)
    keys = sorted(predictions)
    method_order = {"full": 0, "lora": 1, "head_only": 2, "tfidf": 3}
    for left, right in combinations(keys, 2):
        left_method, left_seed = left
        right_method, right_seed = right
        if left_method == right_method:
            continue
        if "tfidf" not in {left_method, right_method} and left_seed != right_seed:
            continue
        if left_method == "tfidf":
            reference, challenger = left, right
        elif right_method == "tfidf":
            reference, challenger = right, left
        elif method_order.get(left_method, 99) < method_order.get(right_method, 99):
            challenger, reference = left, right
        else:
            challenger, reference = right, left
        reference_evidence = predictions[reference]
        challenger_evidence = predictions[challenger]
        if (reference_evidence.example_ids is None) != (challenger_evidence.example_ids is None):
            raise ValueError("paired comparison example IDs are not aligned")
        if reference_evidence.example_ids is not None and not np.array_equal(
            reference_evidence.example_ids, challenger_evidence.example_ids
        ):
            raise ValueError("paired comparison example IDs are not aligned")
        if not np.array_equal(reference_evidence.targets, challenger_evidence.targets):
            raise ValueError("paired comparison targets are not aligned")
        if (
            reference_evidence.labels is not None
            and challenger_evidence.labels is not None
            and reference_evidence.labels != challenger_evidence.labels
        ):
            raise ValueError("paired comparison label mappings are not aligned")
        reference_targets = reference_evidence.targets
        reference_predictions = reference_evidence.predictions
        challenger_predictions = challenger_evidence.predictions
        pair_labels = np.unique(
            np.concatenate((reference_targets, reference_predictions, challenger_predictions))
        ).tolist()
        groups = None
        if schema_by_key[reference] == 3 and reference_evidence.example_ids is not None:
            groups = np.asarray(
                [
                    str(example_id).rsplit(":", maxsplit=1)[0]
                    for example_id in reference_evidence.example_ids
                ]
            )
        name = f"{challenger[0]}_minus_{reference[0]}"
        comparison = paired_bootstrap_difference(
            reference_targets,
            challenger_predictions,
            reference_predictions,
            labels=pair_labels,
            groups=groups,
            samples=bootstrap_samples,
            seed=challenger[1],
        )
        paired[name].append(
            {
                "challenger_seed": challenger[1],
                "reference_seed": reference[1],
                **comparison,
            }
        )

    record = ExperimentRecord(
        run_id="financial-phrasebank-comparison",
        status="completed",
        data={"split_fingerprint": split_fingerprints.pop()},
        # The aggregate has no standalone TOML configuration; keep the source
        # fingerprint while leaving the configuration fingerprint explicitly null.
        provenance=source_provenance(destination.with_name(".comparison-config.toml")),
        metrics={
            "methods": {
                method: _method_summary(rows) for method, rows in sorted(metrics_by_method.items())
            },
            "paired_comparisons": dict(sorted(paired.items())),
        },
    )
    return record.write(destination)
