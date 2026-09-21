import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from transformer_lab.config_io import effective_config, load_experiment_config
from transformer_lab.experiments.manifests import config_digest
from transformer_lab.experiments.selection import (
    candidate_identity,
    load_selection_protocol,
    validate_candidate_summary,
)


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_v3_candidates.py"
    spec = importlib.util.spec_from_file_location("run_v3_candidates_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load candidate runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> object:
    root = Path(__file__).resolve().parents[2]
    return load_experiment_config(root / "configs/sentiment/lora-gpt2-small.toml")


def _candidate(**overrides) -> dict:
    candidate = {
        "candidate_id": "lora-fixture",
        "method": "lora",
        "learning_rate": 0.001,
        "order": 0,
    }
    candidate.update(overrides)
    return candidate


def _completed_summary(config, candidate, *, selection_seed=17, artifact_path=None, digest=None):
    identity = candidate_identity(config, candidate, selection_seed=selection_seed)
    return {
        "schema_version": 1,
        "kind": "sentiment-candidate",
        "status": "completed",
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "effective_config": effective_config(identity),
        "config_digest": config_digest(identity),
        "validation": {"macro_f1": 0.8, "loss": 0.2},
        "artifacts": {
            "model": {
                "path": artifact_path or "model.pt",
                "sha256": digest or ("0" * 64),
            }
        },
    }


def test_summary_rejects_changed_selection_seed():
    config = _config()
    candidate = _candidate()
    summary = _completed_summary(config, candidate)
    changed = candidate_identity(config, candidate, selection_seed=23)

    with pytest.raises(ValueError, match="configuration"):
        validate_candidate_summary(summary, candidate=candidate, identity=changed)


def test_candidate_identity_rejects_invalid_seed_and_learning_rate():
    config = _config()
    candidate = _candidate()

    with pytest.raises(ValueError, match="selection seed"):
        candidate_identity(config, candidate, selection_seed=True)
    with pytest.raises(ValueError, match="positive"):
        candidate_identity(config, {**candidate, "learning_rate": 0}, selection_seed=17)


def test_candidate_summary_rejects_invalid_terminal_evidence(tmp_path):
    config = _config()
    candidate = _candidate()
    identity = candidate_identity(config, candidate, selection_seed=17)

    invalid_status = _completed_summary(config, candidate)
    invalid_status["status"] = "running"
    with pytest.raises(ValueError, match="terminal"):
        validate_candidate_summary(invalid_status, candidate=candidate, identity=identity)

    missing_validation = _completed_summary(config, candidate)
    missing_validation.pop("validation")
    with pytest.raises(ValueError, match="validation metrics"):
        validate_candidate_summary(missing_validation, candidate=candidate, identity=identity)

    missing_model = _completed_summary(config, candidate)
    missing_model["artifacts"] = {}
    with pytest.raises(ValueError, match="model descriptor"):
        validate_candidate_summary(missing_model, candidate=candidate, identity=identity)

    missing_file = _completed_summary(config, candidate, artifact_path="missing.pt")
    with pytest.raises(ValueError, match="missing or outside"):
        validate_candidate_summary(
            missing_file,
            candidate=candidate,
            identity=identity,
            artifact_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda candidate, config: ({**candidate, "candidate_id": "changed"}, config),
            "candidate_id",
        ),
        (lambda candidate, config: ({**candidate, "method": "full"}, config), "method"),
        (
            lambda candidate, config: (
                {**candidate, "learning_rate": 0.002},
                config,
            ),
            "configuration",
        ),
        (
            lambda candidate, config: (
                candidate,
                replace(config, data={**dict(config.data), "dataset_revision": "changed"}),
            ),
            "configuration",
        ),
        (
            lambda candidate, config: (
                candidate,
                replace(config, data={**dict(config.data), "split_seed": 23}),
            ),
            "configuration",
        ),
        (
            lambda candidate, config: (
                candidate,
                replace(config, data={**dict(config.data), "max_length": 32}),
            ),
            "configuration",
        ),
    ],
)
def test_summary_rejects_changed_candidate_identity(change, message):
    config = _config()
    candidate = _candidate()
    summary = _completed_summary(config, candidate)
    changed_candidate, changed_config = change(candidate, config)
    if changed_candidate.get("method") != config.model["mode"]:
        with pytest.raises(ValueError, match=message):
            changed_identity = candidate_identity(
                changed_config, changed_candidate, selection_seed=17
            )
        return
    changed_identity = candidate_identity(changed_config, changed_candidate, selection_seed=17)

    with pytest.raises(ValueError, match=message):
        validate_candidate_summary(summary, candidate=changed_candidate, identity=changed_identity)


def test_completed_summary_accepts_a_matching_model_hash(tmp_path):
    config = _config()
    candidate = _candidate()
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    summary = _completed_summary(config, candidate, artifact_path="model.pt", digest=digest)

    validate_candidate_summary(
        summary,
        candidate=candidate,
        identity=candidate_identity(config, candidate, selection_seed=17),
        artifact_root=tmp_path,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda summary: summary["artifacts"]["model"].update({"sha256": "1" * 64}), "hash"),
        (lambda summary: summary["artifacts"]["model"].update({"path": "../model.pt"}), "escapes"),
        (
            lambda summary: summary["artifacts"]["model"].update({"path": "/absolute/model.pt"}),
            "relative",
        ),
        (
            lambda summary: summary.update({"validation": {"macro_f1": float("nan"), "loss": 0.2}}),
            "macro_f1",
        ),
        (lambda summary: summary.update({"validation": {"macro_f1": 0.8, "loss": -1}}), "loss"),
    ],
)
def test_completed_summary_rejects_corrupt_evidence(tmp_path, mutate, message):
    config = _config()
    candidate = _candidate()
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture model")
    summary = _completed_summary(
        config,
        candidate,
        artifact_path="model.pt",
        digest=hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    mutate(summary)

    with pytest.raises(ValueError, match=message):
        validate_candidate_summary(
            summary,
            candidate=candidate,
            identity=candidate_identity(config, candidate, selection_seed=17),
            artifact_root=tmp_path,
        )


@pytest.mark.parametrize("value", [None, [], "not an object"])
def test_candidate_summary_rejects_malformed_top_level(value):
    config = _config()
    candidate = _candidate()
    with pytest.raises(ValueError, match="object"):
        validate_candidate_summary(
            value,
            candidate=candidate,
            identity=candidate_identity(config, candidate, selection_seed=17),
        )


def test_protocol_summary_cannot_replace_declared_fields(tmp_path):
    source = Path(__file__).resolve().parents[2] / "configs/sentiment/lora-gpt2-small.toml"
    shutil.copyfile(source, tmp_path / "config.toml")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sentiment-candidate",
                "status": "failed",
                "candidate_id": "declared",
                "method": "full",
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.toml"
    protocol.write_text(
        """
[protocol]
name = "fixture"
selection_seed = 17

[[candidates]]
candidate_id = "declared"
method = "lora"
order = 0
config = "config.toml"
summary = "summary.json"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="method"):
        load_selection_protocol(protocol)


def test_runner_rejects_stale_summary_before_training_or_writing(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "configs/sentiment/lora-gpt2-small.toml"
    shutil.copyfile(source, tmp_path / "config.toml")
    summary = tmp_path / "summary.json"
    config = load_experiment_config(tmp_path / "config.toml")
    candidate = _candidate(candidate_id="declared")
    payload = _completed_summary(config, candidate)
    payload["candidate_id"] = "stale"
    original = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    summary.write_text(original, encoding="utf-8")
    protocol = tmp_path / "protocol.toml"
    protocol.write_text(
        """
[protocol]
name = "fixture"
selection_seed = 17

[[candidates]]
candidate_id = "declared"
method = "lora"
order = 0
learning_rate = 0.001
config = "config.toml"
summary = "summary.json"
""",
        encoding="utf-8",
    )
    runner = _runner_module()
    monkeypatch.setattr(
        runner,
        "run_sentiment_candidate",
        lambda *args, **kwargs: pytest.fail("stale evidence must not train"),
    )

    with pytest.raises(ValueError, match="candidate_id"):
        runner.run_protocol(
            protocol,
            device_name="cpu",
            resume=True,
            artifact_root=tmp_path / "artifacts",
        )

    assert summary.read_text(encoding="utf-8") == original
