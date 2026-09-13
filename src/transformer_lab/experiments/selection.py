"""Validation-only selection manifests for sentiment experiments.

Candidate training is deliberately separate from this module.  The selector consumes
validation summaries and produces an auditable frozen choice without loading or
encoding any test examples.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _reject_test_fields(
    value: Any, *, path: str = "root", allowed_keys: frozenset[str] = frozenset()
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if "test" in key_text and key_text not in allowed_keys:
                raise ValueError(f"selection data must not contain test field at {path}.{key}")
            _reject_test_fields(child, path=f"{path}.{key}", allowed_keys=allowed_keys)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _reject_test_fields(child, path=f"{path}[{index}]", allowed_keys=allowed_keys)


def _finite_metric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"completed candidate has invalid validation {name}")
    return float(value)


def select_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select one candidate per method using validation-only deterministic ranking.

    Ranking is descending validation macro-F1, ascending validation loss, then the
    explicit candidate order.  Failed and unavailable candidates remain visible in
    each selected record but cannot be selected.
    """
    if not candidates:
        raise ValueError("selection requires at least one candidate")
    _reject_test_fields(candidates)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for raw in candidates:
        candidate = dict(raw)
        candidate_id = candidate.get("candidate_id")
        method = candidate.get("method")
        order = candidate.get("order")
        status = candidate.get("status")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate selection candidate: {candidate_id}")
        seen_ids.add(candidate_id)
        if not isinstance(method, str) or not method:
            raise ValueError(f"candidate {candidate_id} has no method")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError(f"candidate {candidate_id} has invalid order")
        if status not in {"completed", "failed", "unavailable"}:
            raise ValueError(f"candidate {candidate_id} has invalid status")
        if status == "completed":
            validation = candidate.get("validation")
            if not isinstance(validation, Mapping):
                raise ValueError(f"completed candidate {candidate_id} has no validation metrics")
            candidate["_rank"] = (
                -_finite_metric(validation.get("macro_f1"), "macro_f1"),
                _finite_metric(validation.get("loss"), "loss"),
                order,
            )
        grouped[method].append(candidate)

    selected: dict[str, dict[str, Any]] = {}
    for method, method_candidates in grouped.items():
        completed = [
            candidate for candidate in method_candidates if candidate["status"] == "completed"
        ]
        if not completed:
            selected[method] = {
                "status": "unavailable",
                "candidate_id": None,
                "considered_candidates": [
                    candidate["candidate_id"] for candidate in method_candidates
                ],
                "candidates": method_candidates,
                "error": {"type": "SelectionError", "message": "no completed candidate"},
            }
            continue
        winner = min(completed, key=lambda candidate: candidate["_rank"])
        winner = {key: value for key, value in winner.items() if key != "_rank"}
        validation = winner["validation"]
        selected[method] = {
            "status": "completed",
            "candidate_id": winner["candidate_id"],
            "config_digest": winner.get("config_digest"),
            "config": winner.get("config"),
            "result": winner.get("result"),
            "overrides": {
                "optimization": {
                    key: winner[key] for key in ("learning_rate", "c") if key in winner
                }
            },
            "selection_metric": {
                "validation_macro_f1": float(validation["macro_f1"]),
                "validation_loss": float(validation["loss"]),
            },
            "considered_candidates": [candidate["candidate_id"] for candidate in method_candidates],
            "candidates": method_candidates,
        }
    return selected


def write_selection_manifest(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a schema-checked selection manifest without test data."""
    _reject_test_fields(protocol, path="protocol")
    selection = select_candidates(candidates)
    payload = {
        "schema_version": 1,
        "kind": "sentiment-selection",
        "protocol": dict(protocol),
        "candidates": [dict(candidate) for candidate in candidates],
        "selection": selection,
        "guarantees": {
            "selection_data": "validation_only",
            "test_access": False,
            "ranking": ["validation_macro_f1_desc", "validation_loss_asc", "candidate_order_asc"],
        },
    }
    _reject_test_fields(payload["protocol"], path="protocol")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def load_selection_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a frozen validation-only selection manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid selection manifest: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("selection manifest schema_version must be 1")
    if payload.get("kind") != "sentiment-selection":
        raise ValueError("selection manifest kind must be sentiment-selection")
    _reject_test_fields(payload, allowed_keys=frozenset({"test_access"}))
    if payload.get("guarantees", {}).get("test_access") is not False:
        raise ValueError("selection manifest must declare test_access=false")
    if payload.get("guarantees", {}).get("selection_data") != "validation_only":
        raise ValueError("selection manifest must declare validation_only data")
    selection = payload.get("selection")
    if not isinstance(selection, dict) or not selection:
        raise ValueError("selection manifest has no selected methods")
    return payload


def load_selection_protocol(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a TOML protocol and candidate validation summaries.

    Candidate summaries are intentionally separate from final result records.  A
    summary may contain validation metrics and resource/failure information, but
    any test-labelled field is rejected before it can enter the manifest.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid selection protocol: {path}") from error
    if set(raw) != {"protocol", "candidates"}:
        raise ValueError("selection protocol must contain only protocol and candidates")
    protocol = raw["protocol"]
    candidates = raw["candidates"]
    if not isinstance(protocol, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("selection protocol requires protocol and at least one candidate")
    _reject_test_fields(protocol, path="protocol")
    _reject_test_fields(candidates, path="candidates")
    if not isinstance(protocol.get("name"), str) or not protocol["name"]:
        raise ValueError("selection protocol.name must be a non-empty string")
    prepared: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("selection candidates must be tables")
        item = dict(candidate)
        summary_path = item.pop("summary", None)
        if summary_path is not None:
            if not isinstance(summary_path, str):
                raise ValueError("candidate.summary must be a path string")
            summary_file = Path(summary_path)
            if not summary_file.is_absolute():
                summary_file = (path.parent / summary_file).resolve()
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid candidate summary: {summary_file}") from error
            _reject_test_fields(summary, path=f"candidate[{item.get('candidate_id')}].summary")
            if not isinstance(summary, dict):
                raise ValueError("candidate summary must be an object")
            item.update(summary)
        if "validation" not in item:
            item["validation"] = {
                "macro_f1": item.pop("validation_macro_f1", None),
                "loss": item.pop("validation_loss", None),
            }
        item.setdefault("status", "completed")
        prepared.append(item)
    return protocol, prepared
