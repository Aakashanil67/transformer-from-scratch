from pathlib import Path

import pytest

from transformer_lab.commands import sample, train_lm

pytestmark = pytest.mark.integration


def test_tiny_language_model_workflow_creates_and_reads_a_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_bytes(b"abc" * 30)
    (tmp_path / "validation.txt").write_bytes(b"abc" * 30)
    config = tmp_path / "config.toml"
    config.write_text(
        """
[experiment]
name = "integration-lm"
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
block_size = 2
learning_rate = 0.1
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

    result = train_lm(config, device_name="cpu", seed=3)
    generated = sample(config, device_name="cpu", seed=3)

    assert result.exists()
    assert generated.startswith("a")
