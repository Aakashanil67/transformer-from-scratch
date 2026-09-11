from pathlib import Path

import pytest
import torch

from transformer_lab.config_io import ExperimentConfig
from transformer_lab.experiments.sentiment import run_sentiment_experiment

pytestmark = pytest.mark.integration


def test_sentiment_workflow_records_cpu_preflight_without_claiming_a_score(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment={"name": "integration-sentiment", "kind": "sentiment"},
        model={"mode": "lora"},
        data={},
        optimization={},
        evaluation={},
        output={"result": str(tmp_path / "result.json")},
        generation={},
        source_path=tmp_path / "config.toml",
    )

    result = run_sentiment_experiment(config, device=torch.device("cpu"))

    assert '"status": "unavailable"' in result.read_text(encoding="utf-8")
