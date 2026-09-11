from pathlib import Path
from types import MappingProxyType

import pytest

from transformer_lab.config_io import ExperimentConfig, effective_config, load_experiment_config


def test_load_experiment_config_resolves_paths_and_round_trips(tmp_path: Path) -> None:
    config_file = tmp_path / "example.toml"
    config_file.write_text(
        """
[experiment]
name = "tiny"
kind = "lm"

[model]
model_type = "bigram"

[data]
train = "data/train.txt"
validation = "data/validation.txt"

[optimization]
steps = 4
batch_size = 2
block_size = 3
learning_rate = 0.01

[evaluation]
eval_interval = 2
eval_batches = 1

[output]
directory = "runs/tiny"
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_file)

    assert isinstance(config, ExperimentConfig)
    assert config.data["train"] == str(tmp_path / "data" / "train.txt")
    assert effective_config(config)["experiment"]["kind"] == "lm"
    assert str(tmp_path) not in str(effective_config(config))


def test_load_experiment_config_rejects_unknown_top_level_section(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.toml"
    config_file.write_text(
        "[experiment]\nname='x'\nkind='lm'\n[unknown]\nvalue=1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown section 'unknown'"):
        load_experiment_config(config_file)


def test_load_experiment_config_rejects_invalid_kind(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.toml"
    config_file.write_text("[experiment]\nname='x'\nkind='unknown'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="kind"):
        load_experiment_config(config_file)


def test_loaded_configuration_cannot_be_mutated(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[experiment]\nname='x'\nkind='lm'\n", encoding="utf-8")

    config = load_experiment_config(config_file)

    assert isinstance(config.experiment, MappingProxyType)
    with pytest.raises(TypeError):
        config.experiment["name"] = "changed"


def test_effective_config_keeps_parent_components_without_absolute_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "lm"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(
        "[experiment]\nname='x'\nkind='lm'\n[data]\ntrain='../../data/train.txt'\n",
        encoding="utf-8",
    )

    config = load_experiment_config(config_file)

    assert effective_config(config)["data"]["train"] == "../../data/train.txt"


def test_sentiment_configuration_rejects_unknown_mode(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[experiment]\nname='x'\nkind='sentiment'\n[model]\nmode='magic'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.mode"):
        load_experiment_config(config_file)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("optimization", "batch_size", "0"),
        ("optimization", "gradient_accumulation_steps", "0"),
        ("model", "lora_rank", "0"),
        ("data", "max_length", "0"),
        ("generation", "temperature", "0.0"),
        ("generation", "top_k", "0"),
    ],
)
def test_configuration_rejects_nonpositive_numeric_fields(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f"[experiment]\nname='x'\nkind='lm'\n[{section}]\n{field}={value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_experiment_config(config_file)
