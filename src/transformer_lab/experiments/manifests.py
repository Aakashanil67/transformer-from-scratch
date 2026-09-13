"""Canonical run identities and read-only result verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformer_lab.config_io import ExperimentConfig, effective_config
from transformer_lab.experiments.runtime import file_sha256

_SEED_SUFFIX = re.compile(r"(?:^|-)(?:seed-\d+)$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _without_seed_suffix(value: str) -> str:
    return _SEED_SUFFIX.sub("", value)


def _effective_seed(config: ExperimentConfig, seed: int | None) -> int:
    if seed is not None:
        candidate: Any = seed
    else:
        candidate = config.evaluation.get(
            "seed",
            config.generation.get("seed", config.data.get("seed", 17)),
        )
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError("run seed must be an integer")
    return int(candidate)


def _seeded_sentiment(config: ExperimentConfig) -> bool:
    return config.experiment.get("kind") == "sentiment" and config.model.get("mode") != "baseline"


def resolve_run(config: ExperimentConfig, *, seed: int | None = None) -> ExperimentConfig:
    """Return an idempotently resolved configuration for one effective run."""
    run_seed = _effective_seed(config, seed)
    configured_result = str(config.output.get("result", ""))
    configured_name = str(config.experiment["name"])
    has_seed_identity = bool(
        _SEED_SUFFIX.search(Path(configured_result).stem) or _SEED_SUFFIX.search(configured_name)
    )
    seeded = _seeded_sentiment(config) and (seed is not None or has_seed_identity)
    base_name = _without_seed_suffix(str(config.experiment["name"]))
    run_name = f"{base_name}-seed-{run_seed}" if seeded else base_name

    data = dict(config.data)
    evaluation = dict(config.evaluation)
    output = dict(config.output)
    if seeded:
        data["split_seed"] = int(data.get("split_seed", 17))
        evaluation["seed"] = run_seed
        result_path = Path(str(output.get("result", "")))
        if not str(result_path):
            raise ValueError("output.result must be set")
        result_stem = _without_seed_suffix(result_path.stem)
        result_name = f"{result_stem}-seed-{run_seed}" if result_stem else f"seed-{run_seed}"
        output["result"] = str(result_path.with_name(f"{result_name}{result_path.suffix}"))
        directory = Path(str(output.get("directory", "artifacts/sentiment")))
        output["directory"] = str(
            directory.with_name(f"{_without_seed_suffix(directory.name)}-seed-{run_seed}")
        )
    return replace(
        config,
        experiment={**dict(config.experiment), "name": run_name},
        data=data,
        evaluation=evaluation,
        output=output,
    )


def config_digest(config: ExperimentConfig) -> str:
    """Hash the canonical JSON representation of a resolved configuration."""
    encoded = json.dumps(
        effective_config(resolve_run(config)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    config: ExperimentConfig,
    *,
    provenance: dict[str, Any],
    data_identity: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the identity section shared by new result records."""
    resolved = resolve_run(config)
    return {
        "schema_version": 3,
        "config_sha256": config_digest(resolved),
        "effective_config": effective_config(resolved),
        "data": data_identity or {},
        "evaluation": evaluation or {},
        "artifacts": artifacts or {},
        "provenance": provenance,
    }


def verify_run(
    result_path: Path,
    *,
    artifact_root: Path,
    expected_config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Validate a completed result and every declared file hash without mutation."""
    artifact_root = artifact_root.expanduser().resolve()
    if not result_path.exists():
        raise FileNotFoundError(f"result record does not exist: {result_path}")
    try:
        record = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"result record is not valid JSON: {result_path}") from error
    if not isinstance(record, dict):
        raise ValueError("result record must contain a JSON object")
    schema_version = record.get("schema_version", 2)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("result schema_version must be an integer")
    if schema_version not in {2, 3}:
        raise ValueError(f"unsupported result schema_version: {schema_version}")
    status = record.get("status")
    if status != "completed":
        raise ValueError(f"result status is not reusable: {status!r}")
    if not isinstance(record.get("metrics"), dict):
        raise ValueError("completed result must contain metrics")

    if schema_version == 3:
        manifest = record.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
            raise ValueError("schema-v3 completed result must declare a schema-v3 manifest")
        manifest_config = manifest.get("effective_config")
        record_config = record.get("config")
        manifest_digest = manifest.get("config_sha256")
        if not isinstance(manifest_config, dict) or not isinstance(record_config, dict):
            raise ValueError("schema-v3 manifest and result must declare effective configuration")
        if manifest_config != record_config:
            raise ValueError("result configuration disagrees with its manifest")
        if not isinstance(manifest_digest, str) or not _HASH.fullmatch(manifest_digest):
            raise ValueError("schema-v3 manifest has no valid config_sha256")
        if _payload_digest(manifest_config) != manifest_digest:
            raise ValueError("schema-v3 configuration digest does not match its manifest")
        for field in ("data", "evaluation", "artifacts", "provenance"):
            if not isinstance(manifest.get(field), dict):
                raise ValueError(f"schema-v3 manifest has no {field} identity")

    if expected_config is not None:
        expected = effective_config(resolve_run(expected_config))
        observed = record.get("config")
        if schema_version == 2 and isinstance(observed, dict):
            observed = json.loads(json.dumps(observed))
            expected = json.loads(json.dumps(expected))
            observed.get("experiment", {}).pop("name", None)
            expected.get("experiment", {}).pop("name", None)
        if observed != expected:
            raise ValueError("result configuration does not match the requested configuration")

    artifact_paths: dict[str, Path] = {}
    artifacts = record.get("artifacts", {})
    if artifacts is None:
        artifacts = {}
    if not isinstance(artifacts, dict):
        raise ValueError("result artifacts must be an object")
    if schema_version == 3 and not artifacts:
        raise ValueError("schema-v3 completed result must declare artifacts")
    if schema_version == 3 and record["manifest"].get("artifacts") != artifacts:
        raise ValueError("result artifacts disagree with its manifest")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"artifact descriptor for {name} must be an object")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError(f"artifact {name} has no valid sha256")
        relative = descriptor.get("path")
        candidates = [relative] if isinstance(relative, str) else []
        if relative is None:
            candidates.extend(
                {
                    "model": ("model.pt", "model.joblib"),
                    "evaluation": ("evaluation.pt", "evaluation.joblib"),
                }.get(name, ())
            )
        if not candidates:
            raise ValueError(f"artifact {name} has no verifiable path")
        path = Path(candidates[0])
        for candidate in candidates:
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = artifact_root / candidate_path
            if candidate_path.is_file():
                path = candidate_path
                break
        if not path.is_absolute():
            path = artifact_root / path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"artifact {name} does not exist: {path}")
        observed = file_sha256(path)
        if observed != digest:
            raise ValueError(f"artifact {name} hash does not match the result record")
        artifact_paths[name] = path

    guarantees = ["status", "metrics"]
    if schema_version == 2:
        guarantees.append("schema-v2 compatibility only; v3 identity guarantees are unavailable")
    else:
        guarantees.extend(["configuration" if expected_config is not None else "record identity"])
        guarantees.append("artifact hashes")
    return {
        "record": record,
        "schema_version": schema_version,
        "artifact_paths": artifact_paths,
        "guarantees": guarantees,
    }
