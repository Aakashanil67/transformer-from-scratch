import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformer_lab.config import GPTConfig
from transformer_lab.config_io import ExperimentConfig
from transformer_lab.data.financial_phrasebank import Example, PhraseBankSplits
from transformer_lab.experiments import baseline, parity
from transformer_lab.experiments.sentiment import evaluate_sentiment_run


def _config(
    tmp_path: Path, *, name: str, model: dict[str, object], output: dict[str, object]
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment={"name": name, "kind": "sentiment" if name == "baseline" else "gpt2"},
        model=model,
        data={"dataset_revision": "fixture", "subset": "75Agree"},
        optimization={},
        evaluation={"seed": 17, "bootstrap_samples": 3},
        output=output,
        generation={},
        source_path=tmp_path / "config.toml",
    )


def test_baseline_runner_records_a_model_and_metrics(tmp_path: Path, monkeypatch) -> None:
    examples = tuple(
        Example(text=f"{label} common token {index}", label=label, group_id=str(index))
        for index, label in enumerate(
            [
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
            ]
        )
    )
    splits = PhraseBankSplits(examples[:6], examples[6:8], examples[8:])
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(baseline, "hf_hub_download", lambda *args, **kwargs: str(archive))
    monkeypatch.setattr(baseline, "load_phrasebank", lambda *args, **kwargs: examples)
    monkeypatch.setattr(baseline, "split_phrasebank", lambda *args, **kwargs: splits)
    config = _config(
        tmp_path,
        name="baseline",
        model={"mode": "baseline"},
        output={"result": str(tmp_path / "result.json"), "directory": str(tmp_path / "model")},
    )

    result_path = baseline.run_baseline(config)

    assert result_path.exists()
    assert (tmp_path / "model" / "model.joblib").exists()
    assert (tmp_path / "model" / "evaluation.joblib").exists()
    assert '"status": "completed"' in result_path.read_text(encoding="utf-8")
    assert evaluate_sentiment_run(config, torch.device("cpu")) == result_path


def test_baseline_candidate_writes_validation_only_summary(tmp_path: Path, monkeypatch) -> None:
    examples = tuple(
        Example(text=f"{label} token {index}", label=label, group_id=str(index))
        for index, label in enumerate(["negative", "neutral", "positive"] * 3)
    )

    class SplitSentinel:
        train = examples[:6]
        validation = examples[6:8]

        @property
        def test(self):
            raise AssertionError("validation-only baseline accessed the test partition")

    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(baseline, "hf_hub_download", lambda *args, **kwargs: str(archive))
    monkeypatch.setattr(baseline, "load_phrasebank", lambda *args, **kwargs: examples)
    monkeypatch.setattr(baseline, "split_phrasebank", lambda *args, **kwargs: SplitSentinel())
    config = _config(
        tmp_path,
        name="baseline",
        model={"mode": "baseline"},
        output={"result": str(tmp_path / "result.json"), "directory": str(tmp_path / "model")},
    )
    summary = baseline.run_baseline(
        config,
        validation_only=True,
        summary_path=tmp_path / "summary.json",
        candidate_id="baseline-candidate",
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["candidate_id"] == "baseline-candidate"
    assert "test" not in payload
    assert payload["validation"]["macro_f1"] >= 0
    assert payload["validation"]["loss"] > 0


def test_evaluation_rejects_metrics_that_do_not_match_saved_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    examples = tuple(
        Example(text=f"{label} common token {index}", label=label, group_id=str(index))
        for index, label in enumerate(["negative", "neutral", "positive"] * 4)
    )
    splits = PhraseBankSplits(examples[:6], examples[6:9], examples[9:])
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(baseline, "hf_hub_download", lambda *args, **kwargs: str(archive))
    monkeypatch.setattr(baseline, "load_phrasebank", lambda *args, **kwargs: examples)
    monkeypatch.setattr(baseline, "split_phrasebank", lambda *args, **kwargs: splits)
    config = _config(
        tmp_path,
        name="baseline",
        model={"mode": "baseline"},
        output={"result": str(tmp_path / "result.json"), "directory": str(tmp_path / "model")},
    )
    result_path = baseline.run_baseline(config)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["metrics"]["test"]["macro_f1"] = 0.0
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not match"):
        evaluate_sentiment_run(config, torch.device("cpu"))


def test_baseline_load_examples_uses_the_requested_revision(monkeypatch, tmp_path: Path) -> None:
    example = Example(text="hello", label="neutral", group_id="group")
    monkeypatch.setattr(
        baseline, "hf_hub_download", lambda *args, **kwargs: str(tmp_path / "archive")
    )
    monkeypatch.setattr(baseline, "load_phrasebank", lambda *args, **kwargs: (example,))

    texts, labels = baseline.load_examples("revision")

    assert (texts, labels) == (["hello"], ["neutral"])


def test_parity_runner_records_prompt_errors_and_agreement(tmp_path: Path, monkeypatch) -> None:
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, prompt, **kwargs):
            return {"input_ids": torch.tensor([[1, 2 if prompt == "A" else 3]])}

        def save_pretrained(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeReference:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def state_dict(self):
            return {"reference": torch.ones(1)}

        def __call__(self, token_ids):
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]]))

    class FakeLocal:
        def __init__(self, config):
            self.config = config

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, token_ids):
            return torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]]), None

        def state_dict(self):
            return {"weight": torch.ones(1)}

    monkeypatch.setattr(parity, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(parity, "AutoModelForCausalLM", FakeReference)
    monkeypatch.setattr(parity, "DecoderOnlyTransformer", FakeLocal)
    monkeypatch.setattr(
        parity,
        "gpt2_small_config",
        lambda: GPTConfig(vocab_size=4, block_size=4, n_layer=1, n_head=1, n_embd=4),
    )
    monkeypatch.setattr(
        parity, "load_huggingface_gpt2_weights", lambda *args: SimpleNamespace(loaded=1)
    )
    config = _config(
        tmp_path,
        name="parity",
        model={"prompts": ["A", "B"], "tolerance": 1e-6},
        output={
            "result": str(tmp_path / "parity.json"),
            "directory": str(tmp_path / "gpt2"),
        },
    )

    result_path = parity.run(config)

    assert result_path.exists()
    assert (tmp_path / "gpt2" / "model.pt").exists()
    assert (tmp_path / "gpt2" / "tokenizer" / "tokenizer.json").exists()
    payload = result_path.read_text(encoding="utf-8")
    assert '"status": "completed"' in payload
    assert '"next_token_agreement": 1.0' in payload


def test_random_parity_cases_are_seeded_and_cover_requested_lengths() -> None:
    first = parity.random_token_cases([1, 4, 9], vocab_size=32, seed=17)
    second = parity.random_token_cases([1, 4, 9], vocab_size=32, seed=17)

    assert [tuple(case.shape) for case in first] == [(1, 1), (1, 4), (1, 9)]
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(int(case.max()) < 32 for case in first)
