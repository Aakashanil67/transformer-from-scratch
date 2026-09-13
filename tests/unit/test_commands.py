import hashlib
import json
from pathlib import Path

import pytest
import torch

from transformer_lab.commands import (
    evaluate_matrix,
    resolve_device,
    run_sentiment_matrix,
    sample,
    select_sentiment,
    train_lm,
)
from transformer_lab.config_io import effective_config, load_experiment_config
from transformer_lab.experiments.manifests import build_manifest, resolve_run
from transformer_lab.experiments.selection import write_selection_manifest


def _write_config(root: Path) -> Path:
    (root / "train.txt").write_bytes(b"ab" * 100)
    (root / "validation.txt").write_bytes(b"ab" * 100)
    config = root / "config.toml"
    config.write_text(
        """
[experiment]
name = "tiny-command"
kind = "lm"
[model]
model_type = "bigram"
vocab_size = 256
[data]
train = "train.txt"
validation = "validation.txt"
[optimization]
steps = 2
batch_size = 2
block_size = 1
learning_rate = 0.1
weight_decay = 0.02
warmup_steps = 1
gradient_accumulation_steps = 2
max_grad_norm = 0.5
amp = false
eval_interval = 1
eval_batches = 1
[output]
checkpoint = "model.pt"
result = "result.json"
[generation]
prompt = "a"
max_new_tokens = 2
""",
        encoding="utf-8",
    )
    return config


def _write_transformer_config(root: Path) -> Path:
    (root / "train.txt").write_bytes(b"abcd" * 100)
    (root / "validation.txt").write_bytes(b"abcd" * 100)
    config = root / "transformer.toml"
    config.write_text(
        """
[experiment]
name = "tiny-transformer-command"
kind = "lm"
[model]
model_type = "transformer"
vocab_size = 256
block_size = 4
n_layer = 1
n_head = 2
n_embd = 8
[data]
train = "train.txt"
validation = "validation.txt"
[optimization]
steps = 4
batch_size = 2
block_size = 4
learning_rate = 0.01
eval_interval = 2
eval_batches = 1
[output]
checkpoint = "transformer.pt"
result = "transformer.json"
[generation]
prompt = "a"
max_new_tokens = 2
do_sample = false
""",
        encoding="utf-8",
    )
    return config


def test_train_lm_and_sample_commands_use_the_same_local_checkpoint(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    result = train_lm(config, device_name="cpu", seed=3)
    generated = sample(config, device_name="cpu", seed=3)

    assert result.exists()
    assert generated.startswith("a")
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "scheduler", "history", "config"} <= checkpoint.keys()
    assert checkpoint["config"]["generation"]["seed"] == 3
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["metrics"]["validation_loss"]
    assert payload["timing"]["tokens_per_second"] > 0
    assert payload["environment"]["device"] == "cpu"


def test_train_lm_and_sample_support_the_raw_transformer(tmp_path: Path) -> None:
    config = _write_transformer_config(tmp_path)

    result = train_lm(config, device_name="cpu", seed=3)
    generated = sample(config, device_name="cpu", seed=3)

    assert result.exists()
    assert generated.startswith("a")
    checkpoint = torch.load(tmp_path / "transformer.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_type"] == "transformer"
    assert checkpoint["architecture"]["n_layer"] == 1


def test_auto_device_resolves_to_cpu_when_cuda_is_unavailable() -> None:
    assert str(resolve_device("auto")) == "cpu"


def test_cuda_request_is_explicit_when_unavailable() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_train_lm_rejects_a_missing_prepared_corpus(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    (tmp_path / "train.txt").unlink()

    with pytest.raises(FileNotFoundError, match="prepared"):
        train_lm(config, device_name="cpu")


def test_sample_rejects_a_missing_checkpoint(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        sample(config, device_name="cpu")


def test_matrix_resume_refuses_a_damaged_completed_artifact(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.toml"
    config_path.write_text(
        """
[experiment]
name = "matrix-lora"
kind = "sentiment"
[model]
mode = "lora"
[output]
directory = "run"
result = "result-seed-17.json"
""",
        encoding="utf-8",
    )
    config = resolve_run(load_experiment_config(config_path), seed=17)
    artifact_root = Path(config.output["directory"])
    artifact_root.mkdir()
    model = artifact_root / "model.pt"
    evaluation = artifact_root / "evaluation.pt"
    model.write_bytes(b"model")
    evaluation.write_bytes(b"evaluation")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = Path(config.output["result"])
    artifacts = {
        "model": {"path": "model.pt", "sha256": digest(model)},
        "evaluation": {"path": "evaluation.pt", "sha256": digest(evaluation)},
    }
    result.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": config.experiment["name"],
                "status": "completed",
                "config": effective_config(config),
                "artifacts": artifacts,
                "manifest": build_manifest(
                    config,
                    provenance={},
                    data_identity={},
                    evaluation={},
                    artifacts=artifacts,
                ),
                "metrics": {"test": {"macro_f1": 0.5, "accuracy": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    evaluation.write_bytes(b"damaged")

    with pytest.raises(ValueError, match="evaluation"):
        run_sentiment_matrix(
            [config_path],
            seeds=[17],
            device_name="cpu",
            output=tmp_path / "comparison.json",
            resume=True,
        )

    assert evaluation.read_bytes() == b"damaged"


def test_select_sentiment_writes_a_frozen_manifest_from_validation_summaries(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.toml"
    protocol.write_text(
        """
[protocol]
name = "fixture"
split_seed = 17

[[candidates]]
candidate_id = "head-1"
method = "head_only"
order = 0
config = "config.toml"
validation_macro_f1 = 0.7
validation_loss = 0.4
""",
        encoding="utf-8",
    )

    result = select_sentiment(protocol, output=tmp_path / "selection.json")

    assert result.exists()
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["guarantees"]["test_access"] is False
    assert payload["selection"]["head_only"]["config"] == str(tmp_path / "config.toml")


def test_evaluate_matrix_verifies_selection_before_dispatching(monkeypatch, tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    config = tmp_path / "config.toml"
    write_selection_manifest(
        selection,
        protocol={"name": "fixture"},
        candidates=[
            {
                "candidate_id": "head-1",
                "method": "head_only",
                "order": 0,
                "status": "completed",
                "validation": {"macro_f1": 0.7, "loss": 0.4},
                "config": str(config),
                "learning_rate": 0.002,
            }
        ],
    )
    expected = tmp_path / "comparison.json"
    captured: dict[str, object] = {}

    def fake_matrix(config_paths, **kwargs):
        captured["configs"] = config_paths
        captured.update(kwargs)
        return expected

    monkeypatch.setattr("transformer_lab.commands.run_sentiment_matrix", fake_matrix)

    result = evaluate_matrix(
        selection,
        seeds=[17],
        device_name="cpu",
        output=expected,
    )

    assert result == expected
    assert captured["configs"] == [config]
    assert captured["seeds"] == [17]
    assert captured["config_overrides"] == {str(config): {"optimization": {"learning_rate": 0.002}}}


def test_evaluate_matrix_rejects_unavailable_and_stale_selected_candidates(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    unavailable = tmp_path / "unavailable.json"
    write_selection_manifest(
        unavailable,
        protocol={"name": "fixture"},
        candidates=[
            {
                "candidate_id": "head-unavailable",
                "method": "head_only",
                "order": 0,
                "status": "unavailable",
            }
        ],
    )
    with pytest.raises(RuntimeError, match="no completed"):
        evaluate_matrix(unavailable, seeds=[17], device_name="cpu", output=tmp_path / "out.json")

    stale = tmp_path / "stale.json"
    write_selection_manifest(
        stale,
        protocol={"name": "fixture"},
        candidates=[
            {
                "candidate_id": "head-stale",
                "method": "head_only",
                "order": 0,
                "status": "completed",
                "validation": {"macro_f1": 0.7, "loss": 0.4},
                "config": str(config),
                "config_digest": "0" * 64,
            }
        ],
    )
    with pytest.raises(ValueError, match="stale"):
        evaluate_matrix(stale, seeds=[17], device_name="cpu", output=tmp_path / "out.json")
