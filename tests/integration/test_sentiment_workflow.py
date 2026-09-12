from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformer_lab.config import GPTConfig
from transformer_lab.config_io import ExperimentConfig
from transformer_lab.experiments.sentiment import _train_transformer, run_sentiment_experiment
from transformer_lab.models.sentiment import SentimentClassifier

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_training_resume_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    device = torch.device("cuda")
    architecture = GPTConfig(
        vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8, dropout=0.0
    )
    torch.manual_seed(11)
    initial = SentimentClassifier(architecture, num_labels=3).state_dict()
    inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 0, 0], [8, 9, 10, 0]])
    masks = (inputs != 0).long()
    labels = torch.tensor([0, 1, 2, 1])

    def model() -> SentimentClassifier:
        instance = SentimentClassifier(architecture, num_labels=3)
        instance.load_state_dict(initial)
        return instance.to(device)

    def loader() -> DataLoader:
        return DataLoader(
            TensorDataset(inputs, masks, labels),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(17),
        )

    uninterrupted = model()
    _train_transformer(
        uninterrupted,
        loader(),
        loader(),
        device=device,
        epochs=2,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=1,
        max_grad_norm=1,
        amp=False,
    )
    checkpoint = tmp_path / "training.pt"
    best = tmp_path / "best.pt"
    staged = model()
    _train_transformer(
        staged,
        loader(),
        loader(),
        device=device,
        epochs=1,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=1,
        max_grad_norm=1,
        amp=False,
        checkpoint_path=checkpoint,
        best_model_path=best,
        checkpoint_config={"seed": 17},
    )
    resumed = model()
    _train_transformer(
        resumed,
        loader(),
        loader(),
        device=device,
        epochs=2,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=1,
        max_grad_norm=1,
        amp=False,
        checkpoint_path=checkpoint,
        best_model_path=best,
        checkpoint_config={"seed": 17},
        resume=True,
    )

    for name, expected in uninterrupted.state_dict().items():
        assert torch.allclose(resumed.state_dict()[name], expected, atol=1e-6), name
