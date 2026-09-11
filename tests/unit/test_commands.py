import json
from pathlib import Path

import pytest
import torch

from transformer_lab.commands import resolve_device, sample, train_lm


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


def test_train_lm_and_sample_commands_use_the_same_local_checkpoint(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    result = train_lm(config, device_name="cpu", seed=3)
    generated = sample(config, device_name="cpu", seed=3)

    assert result.exists()
    assert generated.startswith("a")
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "scheduler", "history", "config"} <= checkpoint.keys()
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["metrics"]["validation_loss"]
    assert payload["timing"]["tokens_per_second"] > 0
    assert payload["environment"]["device"] == "cpu"


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
