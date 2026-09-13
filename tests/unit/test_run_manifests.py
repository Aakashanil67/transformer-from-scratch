import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from transformer_lab.config_io import ExperimentConfig, effective_config
from transformer_lab.experiments.manifests import build_manifest, resolve_run, verify_run


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment={"name": "phrasebank-lora", "kind": "sentiment"},
        model={
            "mode": "lora",
            "base_model": "gpt2",
            "model_revision": "revision-a",
            "lora_rank": 4,
            "num_labels": 3,
        },
        data={"dataset_revision": "data-a", "subset": "75Agree", "max_length": 64},
        optimization={"learning_rate": 0.001, "epochs": 3},
        evaluation={"seed": 17},
        output={
            "directory": str(tmp_path / "artifacts" / "phrasebank"),
            "result": str(tmp_path / "results" / "phrasebank-seed-17.json"),
        },
        generation={},
        source_path=tmp_path / "config.toml",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completed_result(config: ExperimentConfig, artifact_root: Path) -> Path:
    model = artifact_root / "model.pt"
    evaluation = artifact_root / "evaluation.pt"
    artifact_root.mkdir(parents=True)
    model.write_bytes(b"model")
    evaluation.write_bytes(b"evaluation")
    resolved = resolve_run(config, seed=17)
    result = Path(resolved.output["result"])
    result.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "model": {"path": "model.pt", "sha256": _sha256(model)},
        "evaluation": {"path": "evaluation.pt", "sha256": _sha256(evaluation)},
    }
    result.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": resolved.experiment["name"],
                "status": "completed",
                "config": effective_config(resolved),
                "artifacts": artifacts,
                "manifest": build_manifest(
                    resolved,
                    provenance={},
                    data_identity={},
                    evaluation={},
                    artifacts=artifacts,
                ),
                "metrics": {"test": {"macro_f1": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    return result


def test_resolve_run_is_idempotent_and_matches_matrix_seed_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)

    direct = resolve_run(config, seed=17)
    matrix = resolve_run(config, seed=17)
    repeated = resolve_run(direct, seed=17)
    other_seed = resolve_run(config, seed=23)

    assert direct.output == matrix.output
    assert repeated.output == direct.output
    assert repeated.experiment == direct.experiment
    assert direct.experiment["name"] == "phrasebank-lora-seed-17"
    assert other_seed.output["result"] != direct.output["result"]
    assert "seed-17-seed-17" not in str(repeated.output["result"])


def test_verify_run_rejects_configuration_and_artifact_mismatches(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifact_root = tmp_path / "artifacts" / "phrasebank-seed-17"
    result = _write_completed_result(config, artifact_root)

    verify_run(result, artifact_root=artifact_root, expected_config=config)

    changed = replace(config, optimization={"learning_rate": 0.002, "epochs": 3})
    with pytest.raises(ValueError, match="configuration"):
        verify_run(result, artifact_root=artifact_root, expected_config=changed)

    (artifact_root / "evaluation.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evaluation"):
        verify_run(result, artifact_root=artifact_root)


def test_verify_run_rejects_a_drifted_manifest_digest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifact_root = tmp_path / "artifacts" / "phrasebank-seed-17"
    result = _write_completed_result(config, artifact_root)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["manifest"]["effective_config"]["model"]["lora_rank"] = 8
    payload["config"] = payload["manifest"]["effective_config"]
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        verify_run(result, artifact_root=artifact_root)


@pytest.mark.parametrize("status", ["failed", "unavailable"])
def test_verify_run_rejects_non_completed_statuses(tmp_path: Path, status: str) -> None:
    config = resolve_run(_config(tmp_path), seed=17)
    result = Path(config.output["result"])
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": config.experiment["name"],
                "status": status,
                "metrics": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="status"):
        verify_run(result, artifact_root=tmp_path)


def test_verify_run_rejects_missing_and_truncated_results(tmp_path: Path) -> None:
    config = resolve_run(_config(tmp_path), seed=17)
    result = Path(config.output["result"])
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        verify_run(result, artifact_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="result"):
        verify_run(result.with_name("missing.json"), artifact_root=tmp_path)
