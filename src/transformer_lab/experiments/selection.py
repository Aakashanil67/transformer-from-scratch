"""Validation-only selection manifests for sentiment experiments.

Candidate training is deliberately separate from this module.  The selector consumes
validation summaries and produces an auditable frozen choice without loading or
encoding any test examples.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from transformer_lab.config_io import ExperimentConfig, effective_config, load_experiment_config
from transformer_lab.experiments.manifests import config_digest
from transformer_lab.experiments.runtime import file_sha256

_HASH = re.compile(r"^[0-9a-f]{64}$")


def candidate_identity(
    config: ExperimentConfig,
    candidate: Mapping[str, Any],
    *,
    selection_seed: int,
) -> ExperimentConfig:
    """Return the canonical configuration represented by one protocol candidate."""
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection seed must be an integer")
    if candidate.get("method") != config.model.get("mode"):
        raise ValueError("candidate method disagrees with its configuration")
    optimization = dict(config.optimization)
    for key in ("learning_rate", "c"):
        if key in candidate:
            value = candidate[key]
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"candidate {key} must be positive")
            optimization[key] = candidate[key]
    return replace(
        config,
        optimization=optimization,
        evaluation={**dict(config.evaluation), "seed": selection_seed},
    )


def validate_candidate_summary(
    summary: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    identity: ExperimentConfig,
    artifact_root: Path | None = None,
) -> None:
    """Validate validation-only candidate evidence and, when available, its hashes."""
    if not isinstance(summary, Mapping):
        raise ValueError("candidate summary must be an object")
    _reject_test_fields(summary)
    if summary.get("schema_version") != 1 or summary.get("kind") != "sentiment-candidate":
        raise ValueError("unsupported candidate summary schema")
    for key in ("candidate_id", "method"):
        if summary.get(key) != candidate.get(key):
            raise ValueError(f"candidate summary disagrees with declared {key}")
    if summary.get("effective_config") != effective_config(identity):
        raise ValueError("candidate summary has a stale effective configuration")
    if summary.get("config_digest") != config_digest(identity):
        raise ValueError("candidate summary has a stale configuration digest")
    status = summary.get("status")
    if status not in {"completed", "failed", "unavailable"}:
        raise ValueError("candidate summary has no terminal status")
    if status != "completed":
        return
    validation = summary.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("completed candidate has no validation metrics")
    macro_f1 = validation.get("macro_f1")
    loss = validation.get("loss")
    for key, value in (("macro_f1", macro_f1), ("loss", loss)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"candidate validation {key} must be finite")
        if value < 0 or (key == "macro_f1" and value > 1):
            raise ValueError(f"candidate validation {key} is out of range")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or "model" not in artifacts:
        raise ValueError("completed candidate has no model descriptor")
    for descriptor in artifacts.values():
        if not isinstance(descriptor, Mapping):
            raise ValueError("candidate artifact descriptor must be an object")
        digest, relative = descriptor.get("sha256"), descriptor.get("path")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("candidate artifact has no valid sha256")
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path is None
            or relative_path.is_absolute()
            or bool(relative_path.anchor)
            or PurePosixPath(relative).is_absolute()
            or relative.startswith(("/", "\\"))
        ):
            raise ValueError("candidate artifact path must be relative")
        if ".." in relative_path.parts or ".." in PurePosixPath(relative).parts:
            raise ValueError("candidate artifact path escapes its root")
        if artifact_root is None:
            continue
        root = artifact_root.expanduser().resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("candidate artifact is missing or outside its root")
        if file_sha256(path) != digest:
            raise ValueError("candidate artifact hash mismatch")


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
    seen_orders: dict[str, set[int]] = defaultdict(set)
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
        if order in seen_orders[method]:
            raise ValueError(
                f"candidate {candidate_id} duplicates order {order} for method {method}"
            )
        seen_orders[method].add(order)
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
    guarantees = payload.get("guarantees")
    if not isinstance(guarantees, Mapping):
        raise ValueError(
            "selection manifest guarantees must be an object declaring test_access=false"
        )
    if guarantees.get("test_access") is not False:
        raise ValueError("selection manifest must declare test_access=false")
    if guarantees.get("selection_data") != "validation_only":
        raise ValueError("selection manifest must declare validation_only data")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("selection manifest candidates must be a non-empty list")
    selection = payload.get("selection")
    if not isinstance(selection, dict) or not selection:
        raise ValueError("selection manifest has no selected methods")
    try:
        expected = json.loads(json.dumps(select_candidates(candidates), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid selection candidate evidence; validation metrics must be finite: {error}"
        ) from error
    if selection != expected:
        raise ValueError("stored selection disagrees with candidate evidence")
    return payload


def load_selection_protocol(
    path: Path, *, allow_missing_summaries: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    selection_seed = protocol.get("selection_seed", 17)
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection protocol.selection_seed must be an integer")
    prepared: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("selection candidates must be tables")
        declaration = dict(candidate)
        summary_path = declaration.pop("summary", None)
        if summary_path is not None:
            if not isinstance(summary_path, str):
                raise ValueError("candidate.summary must be a path string")
            summary_file = Path(summary_path)
            if not summary_file.is_absolute():
                summary_file = (path.parent / summary_file).resolve()
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                if not allow_missing_summaries:
                    raise ValueError(f"invalid candidate summary: {summary_file}") from error
                item = {**declaration, "summary_path": str(summary_file)}
                item.setdefault("status", "pending")
                _validate_protocol_candidate(path, protocol, item)
                prepared.append(item)
                continue
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid candidate summary: {summary_file}") from error
            if not isinstance(summary, dict):
                raise ValueError("candidate summary must be an object")
            _reject_test_fields(
                summary, path=f"candidate[{declaration.get('candidate_id')}].summary"
            )
            _validate_protocol_candidate(path, protocol, declaration)
            config = _candidate_config(path, declaration)
            if config is not None:
                identity = candidate_identity(config, declaration, selection_seed=selection_seed)
                validate_candidate_summary(summary, candidate=declaration, identity=identity)
            item = _merge_candidate_summary(declaration, summary)
            item["summary_path"] = str(summary_file)
            prepared.append(item)
            continue
        item = declaration
        _validate_protocol_candidate(path, protocol, item)
        if "validation" not in item:
            item["validation"] = {
                "macro_f1": item.pop("validation_macro_f1", None),
                "loss": item.pop("validation_loss", None),
            }
        item.setdefault("status", "completed")
        prepared.append(item)
    return protocol, prepared


def _candidate_config(protocol_path: Path, candidate: Mapping[str, Any]) -> ExperimentConfig | None:
    config_value = candidate.get("config")
    if not isinstance(config_value, str):
        return None
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = (protocol_path.parent / config_path).resolve()
    if not config_path.is_file():
        return None
    return load_experiment_config(config_path)


def _validate_protocol_candidate(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    if "config" not in candidate:
        return
    config_path = Path(str(candidate["config"]))
    if not config_path.is_absolute():
        config_path = (protocol_path.parent / config_path).resolve()
    if not config_path.is_file():
        return
    config = load_experiment_config(config_path)
    if candidate.get("method") != config.model.get("mode"):
        raise ValueError("candidate method disagrees with its configuration")
    common = (
        ("dataset_revision", "data"),
        ("subset", "data"),
        ("split_seed", "data"),
    )
    for key, section in common:
        if key in protocol and config.__getattribute__(section).get(key) != protocol[key]:
            raise ValueError(f"protocol {key} disagrees with candidate configuration")
    if candidate.get("method") == "baseline":
        return
    transformer = (
        ("max_length", "data"),
        ("epochs", "optimization"),
        ("batch_size", "optimization"),
        ("gradient_accumulation_steps", "optimization"),
    )
    for key, section in transformer:
        if key in protocol and config.__getattribute__(section).get(key) != protocol[key]:
            raise ValueError(f"protocol {key} disagrees with candidate configuration")


def _merge_candidate_summary(
    declaration: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    item = dict(declaration)
    for key, value in summary.items():
        if key in declaration:
            if declaration[key] != value:
                raise ValueError(f"candidate summary conflicts with declared {key}")
            continue
        item[key] = value
    return item
