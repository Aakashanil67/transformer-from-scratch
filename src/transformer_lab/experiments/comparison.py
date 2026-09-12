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
    paired_bootstrap_difference,
    summarise_seed_metrics,
)
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import source_provenance


@dataclass(frozen=True)
class RunOutput:
    method: str
    seed: int
    result: Path
    evaluation: Path


def _predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".joblib":
        payload = load(path)
        targets = payload.get("target_ids")
        predictions = payload.get("prediction_ids")
        if targets is None:
            targets = payload["targets"]
        if predictions is None:
            predictions = payload["predictions"]
        return np.asarray(targets), np.asarray(predictions)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return np.asarray(payload["targets"]), np.asarray(payload["predictions"])


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
    predictions = {}
    for run in runs:
        record = json.loads(run.result.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise ValueError(f"comparison input is not completed: {run.result.name}")
        split = record.get("data", {}).get("split", {})
        fingerprint = split.get("sha256", split.get("fingerprint"))
        if not fingerprint:
            raise ValueError(f"comparison input has no split fingerprint: {run.result.name}")
        split_fingerprints.add(str(fingerprint))
        test = record["metrics"]["test"]
        metrics_by_method[run.method].append(
            {"seed": run.seed, "macro_f1": test["macro_f1"], "accuracy": test["accuracy"]}
        )
        predictions[(run.method, run.seed)] = _predictions(run.evaluation)
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
        reference_targets, reference_predictions = predictions[reference]
        challenger_targets, challenger_predictions = predictions[challenger]
        if not np.array_equal(reference_targets, challenger_targets):
            raise ValueError("paired comparison targets are not aligned")
        name = f"{challenger[0]}_minus_{reference[0]}"
        comparison = paired_bootstrap_difference(
            reference_targets,
            challenger_predictions,
            reference_predictions,
            labels=np.unique(reference_targets).tolist(),
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
