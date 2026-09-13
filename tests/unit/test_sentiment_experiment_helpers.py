import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import transformer_lab.experiments.sentiment as sentiment_module
from transformer_lab.config import GPTConfig
from transformer_lab.config_io import ExperimentConfig
from transformer_lab.evaluation.classification import apply_temperature, classification_metrics
from transformer_lab.experiments.sentiment import (
    _encode,
    _metrics_match,
    _predict,
    _train_transformer,
    evaluate_sentiment_run,
    run_sentiment_experiment,
    sentiment_seeds,
)
from transformer_lab.models.sentiment import SentimentClassifier


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        width = kwargs["max_length"]
        ids = []
        masks = []
        for text in texts:
            values = [ord(char) % 10 + 1 for char in text][:width]
            mask = [1] * len(values)
            values.extend([0] * (width - len(values)))
            mask.extend([0] * (width - len(mask)))
            ids.append(values)
            masks.append(mask)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def _config(tmp_path: Path, status_name: str = "sentiment") -> ExperimentConfig:
    return ExperimentConfig(
        experiment={"name": status_name, "kind": "sentiment"},
        model={"mode": "lora"},
        data={},
        optimization={},
        evaluation={"seed": 17},
        output={"result": str(tmp_path / f"{status_name}.json")},
        generation={},
        source_path=tmp_path / "config.toml",
    )


def test_sentiment_helpers_train_and_predict_on_a_tiny_batch() -> None:
    examples = [
        type("Example", (), {"text": text, "label": label})
        for text, label in [("up", "positive"), ("down", "negative")]
    ]
    input_ids, masks, labels = _encode(FakeTokenizer(), examples, 4)
    dataset = TensorDataset(input_ids, masks, labels)
    loader = DataLoader(dataset, batch_size=2)
    model = SentimentClassifier(
        GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8), num_labels=3
    )

    result = _train_transformer(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=1,
        max_grad_norm=1,
        amp=False,
    )
    targets, predictions, probabilities, logits = _predict(model, loader, torch.device("cpu"))

    assert len(result["losses"]) == 1
    assert targets.shape == predictions.shape == (2,)
    assert probabilities.shape == (2, 3)
    assert logits.shape == (2, 3)


def test_predict_restores_training_mode_when_model_evaluation_fails() -> None:
    model = SentimentClassifier(
        GPTConfig(vocab_size=4, block_size=4, n_layer=1, n_head=2, n_embd=8), num_labels=3
    ).train()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[4, 0, 0, 0]]),
            torch.tensor([[1, 0, 0, 0]]),
            torch.tensor([0]),
        ),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="out-of-range"):
        _predict(model, loader, torch.device("cpu"))
    assert model.training


def test_incomplete_accumulation_window_uses_its_actual_batch_count() -> None:
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8)
    model = SentimentClassifier(config, num_labels=3)
    reference = SentimentClassifier(config, num_labels=3)
    reference.load_state_dict(model.state_dict())
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(input_ids, masks, labels), batch_size=2)
    _, reference_loss = reference(input_ids, attention_mask=masks, labels=labels)
    assert reference_loss is not None
    reference_loss.backward()
    expected = reference.classifier.weight.grad.clone()
    observed: list[torch.Tensor] = []
    model.classifier.weight.register_hook(lambda gradient: observed.append(gradient.clone()))

    _train_transformer(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=4,
        max_grad_norm=1,
        amp=False,
    )

    assert torch.allclose(observed[0], expected)


def test_transformer_training_resumes_at_the_next_epoch(tmp_path: Path) -> None:
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8)
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(input_ids, masks, labels), batch_size=2)
    checkpoint = tmp_path / "training.pt"
    best = tmp_path / "best.pt"
    first = _train_transformer(
        SentimentClassifier(config, num_labels=3),
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=1,
        max_grad_norm=1,
        amp=False,
        checkpoint_path=checkpoint,
        best_model_path=best,
        checkpoint_config={"run": "fixture"},
    )
    resumed = _train_transformer(
        SentimentClassifier(config, num_labels=3),
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=0.01,
        weight_decay=0,
        accumulation=1,
        max_grad_norm=1,
        amp=False,
        checkpoint_path=checkpoint,
        best_model_path=best,
        checkpoint_config={"run": "fixture"},
        resume=True,
    )

    assert checkpoint.exists() and best.exists()
    assert first["best_epoch"] == 1
    assert [candidate["epoch"] for candidate in resumed["validation_candidates"]] == [1, 2]


def test_transformer_training_rejects_a_missing_selected_best_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8)
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(input_ids, masks, labels), batch_size=2)
    monkeypatch.setattr(sentiment_module, "_atomic_torch_save", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="best"):
        _train_transformer(
            SentimentClassifier(config, num_labels=3),
            loader,
            loader,
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=0.01,
            weight_decay=0,
            accumulation=1,
            max_grad_norm=1,
            amp=False,
            best_model_path=tmp_path / "best.pt",
        )


def test_cpu_sentiment_preflight_writes_an_unavailable_record(tmp_path: Path) -> None:
    config = _config(tmp_path, "unavailable")

    result = run_sentiment_experiment(config, device=torch.device("cpu"))

    assert result.exists()
    assert '"status": "unavailable"' in result.read_text(encoding="utf-8")


def test_cuda_oom_writes_a_failed_record(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path, "oom")

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory while allocating 1 GiB")

    monkeypatch.setattr(sentiment_module, "_execute_sentiment_experiment", fail)
    result = run_sentiment_experiment(config, device=torch.device("cuda"))
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["metrics"] is None
    assert payload["error"]["type"] == "OutOfMemoryError"


def test_evaluate_sentiment_run_rejects_an_unavailable_record(tmp_path: Path) -> None:
    config = _config(tmp_path, "incomplete")
    run_sentiment_experiment(config, device=torch.device("cpu"))

    try:
        evaluate_sentiment_run(config, torch.device("cpu"))
    except RuntimeError as error:
        assert "not complete" in str(error)
    else:
        raise AssertionError("unavailable run was evaluated")


def test_evaluate_sentiment_run_rejects_a_completed_record_without_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "complete")
    Path(config.output["result"]).write_text(
        json.dumps({"status": "completed", "metrics": {"test": {"accuracy": 1.0}}}),
        encoding="utf-8",
    )

    try:
        evaluate_sentiment_run(config, torch.device("cpu"))
    except FileNotFoundError as error:
        assert "evaluation" in str(error)
    else:
        raise AssertionError("a run without evaluation artifacts was accepted")


def test_evaluate_sentiment_run_replays_a_transformer_checkpoint(tmp_path: Path) -> None:
    architecture = {
        "vocab_size": 16,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "dropout": 0.0,
        "bias": True,
    }
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    model = SentimentClassifier(GPTConfig(**architecture), num_labels=3).eval()
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]])
    masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    targets = torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(input_ids, masks, targets), batch_size=2)
    actual_targets, predictions, _, logits = _predict(model, loader, torch.device("cpu"))
    probabilities = apply_temperature(logits, 1.0)
    metrics = classification_metrics(
        actual_targets,
        predictions,
        labels=[0, 1, 2],
        probabilities=probabilities,
        bootstrap_samples=5,
        seed=17,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "mode": "head_only",
            "num_labels": 3,
        },
        output_dir / "model.pt",
    )
    torch.save(
        {"input_ids": input_ids, "attention_mask": masks, "targets": targets},
        output_dir / "evaluation.pt",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"status": "completed", "metrics": {"test": metrics}}),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        experiment={"name": "complete", "kind": "sentiment"},
        model={"mode": "head_only"},
        data={},
        optimization={},
        evaluation={"seed": 17, "bootstrap_samples": 5},
        output={"result": str(result_path), "directory": str(output_dir)},
        generation={},
        source_path=tmp_path / "config.toml",
    )

    assert evaluate_sentiment_run(config, torch.device("cpu")) == result_path


def test_evaluate_sentiment_run_resolves_seeded_artifact_directory(tmp_path: Path) -> None:
    architecture = {
        "vocab_size": 16,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "dropout": 0.0,
        "bias": True,
    }
    configured_dir = tmp_path / "model"
    seeded_dir = tmp_path / "model-seed-17"
    seeded_dir.mkdir()
    model = SentimentClassifier(GPTConfig(**architecture), num_labels=3).eval()
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]])
    masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    targets = torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(input_ids, masks, targets), batch_size=2)
    actual_targets, predictions, _, logits = _predict(model, loader, torch.device("cpu"))
    probabilities = apply_temperature(logits, 1.0)
    metrics = classification_metrics(
        actual_targets,
        predictions,
        labels=[0, 1, 2],
        probabilities=probabilities,
        bootstrap_samples=5,
        seed=17,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "mode": "head_only",
            "num_labels": 3,
        },
        seeded_dir / "model.pt",
    )
    torch.save(
        {"input_ids": input_ids, "attention_mask": masks, "targets": targets},
        seeded_dir / "evaluation.pt",
    )
    result_path = tmp_path / "result-seed-17.json"
    result_path.write_text(
        json.dumps({"status": "completed", "metrics": {"test": metrics}}),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        experiment={"name": "complete", "kind": "sentiment"},
        model={"mode": "head_only"},
        data={},
        optimization={},
        evaluation={"seed": 17, "bootstrap_samples": 5},
        output={"result": str(result_path), "directory": str(configured_dir)},
        generation={},
        source_path=tmp_path / "config.toml",
    )

    assert evaluate_sentiment_run(config, torch.device("cpu")) == result_path


def test_sentiment_seeds_do_not_change_the_split_seed(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment={"name": "matrix", "kind": "sentiment"},
        model={"mode": "lora"},
        data={"split_seed": 17},
        optimization={},
        evaluation={"seed": 17},
        output={"result": str(tmp_path / "seed-17.json"), "directory": str(tmp_path / "run")},
        generation={},
        source_path=tmp_path / "config.toml",
    )

    runs = sentiment_seeds(config, [17, 23, 41])

    assert [run.data["split_seed"] for run in runs] == [17, 17, 17]
    assert [run.evaluation["seed"] for run in runs] == [17, 23, 41]
    assert [Path(run.output["result"]).name for run in runs] == [
        "seed-17.json",
        "seed-23.json",
        "seed-41.json",
    ]


def test_metric_replay_allows_float_rounding_but_not_metric_drift() -> None:
    expected = {"nested": {"score": 0.2721848367349916}, "count": 518}
    assert _metrics_match(expected, {"nested": {"score": 0.2721849}, "count": 518})
    assert not _metrics_match(expected, {"nested": {"score": 0.2729}, "count": 518})
