from pathlib import Path

import pytest
import torch

import transformer_lab.training as training_module
from transformer_lab.data.batches import sample_batch
from transformer_lab.models.bigram import BigramLanguageModel
from transformer_lab.training import TrainingConfig, _lr_lambda, train_language_model


def test_training_reduces_bigram_validation_loss_on_a_repeating_corpus() -> None:
    torch.manual_seed(11)
    model = BigramLanguageModel(vocab_size=2)
    token_ids = torch.tensor([0, 1] * 100)

    history = train_language_model(
        model,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=TrainingConfig(
            steps=40,
            batch_size=8,
            block_size=1,
            learning_rate=0.1,
            eval_interval=20,
            eval_batches=4,
            seed=11,
        ),
    )

    assert len(history.validation_losses) == 3
    assert history.validation_losses[-1] < history.validation_losses[0]


def test_evaluation_cadence_does_not_change_training_trajectory() -> None:
    token_ids = torch.tensor([0, 1] * 100)
    first = BigramLanguageModel(vocab_size=2)
    second = BigramLanguageModel(vocab_size=2)
    second.load_state_dict(first.state_dict())

    common = dict(
        steps=12,
        batch_size=8,
        block_size=1,
        learning_rate=0.1,
        eval_batches=4,
        seed=11,
    )
    train_language_model(
        first,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=TrainingConfig(eval_interval=4, **common),
    )
    train_language_model(
        second,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=TrainingConfig(eval_interval=6, **common),
    )

    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        )
    )


@pytest.mark.parametrize(
    "field", ["steps", "batch_size", "block_size", "eval_interval", "eval_batches"]
)
def test_training_config_rejects_non_positive_counts(field: str) -> None:
    values = dict(
        steps=1,
        batch_size=1,
        block_size=1,
        learning_rate=0.1,
        eval_interval=1,
        eval_batches=1,
        seed=1,
    )
    values[field] = 0

    with pytest.raises(ValueError, match=field):
        TrainingConfig(**values)


def test_resume_rejects_a_checkpoint_from_a_different_configuration(tmp_path: Path) -> None:
    token_ids = torch.tensor([0, 1] * 20)
    config = TrainingConfig(
        steps=1,
        batch_size=2,
        block_size=1,
        learning_rate=0.1,
        eval_interval=1,
        eval_batches=1,
        seed=7,
    )
    checkpoint = tmp_path / "model.pt"
    train_language_model(
        BigramLanguageModel(vocab_size=2),
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=config,
        checkpoint_path=checkpoint,
        checkpoint_config={"run": 1},
    )

    with pytest.raises(ValueError, match="configuration"):
        train_language_model(
            BigramLanguageModel(vocab_size=2),
            train_token_ids=token_ids,
            validation_token_ids=token_ids,
            config=config,
            checkpoint_path=checkpoint,
            resume=True,
            checkpoint_config={"run": 2},
        )


def test_resume_after_a_complete_update_matches_uninterrupted_accumulated_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_ids = torch.tensor([0, 1, 0, 1, 1, 0] * 40)
    config = TrainingConfig(
        steps=8,
        batch_size=2,
        block_size=2,
        learning_rate=0.05,
        eval_interval=3,
        eval_batches=1,
        seed=17,
        gradient_accumulation_steps=4,
    )
    initial = BigramLanguageModel(vocab_size=2)
    uninterrupted = BigramLanguageModel(vocab_size=2)
    uninterrupted.load_state_dict(initial.state_dict())
    expected_checkpoint = tmp_path / "uninterrupted.pt"
    expected_history = train_language_model(
        uninterrupted,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=config,
        checkpoint_path=expected_checkpoint,
        checkpoint_config={"run": "accumulated"},
    )

    checkpoint = tmp_path / "resume.pt"
    interrupted = BigramLanguageModel(vocab_size=2)
    interrupted.load_state_dict(initial.state_dict())
    real_save = training_module.save_checkpoint

    def save_then_interrupt(*args: object, **kwargs: object) -> Path:
        result = real_save(*args, **kwargs)
        if kwargs["step"] == 4:
            raise RuntimeError("test interruption after checkpoint")
        return result

    monkeypatch.setattr(training_module, "save_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="test interruption"):
        train_language_model(
            interrupted,
            train_token_ids=token_ids,
            validation_token_ids=token_ids,
            config=config,
            checkpoint_path=checkpoint,
            checkpoint_config={"run": "accumulated"},
        )

    resumed = BigramLanguageModel(vocab_size=2)
    resumed.load_state_dict(initial.state_dict())
    resumed_history = train_language_model(
        resumed,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=config,
        checkpoint_path=checkpoint,
        resume=True,
        checkpoint_config={"run": "accumulated"},
    )

    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=1e-6)
    assert resumed_history.training_losses == expected_history.training_losses
    assert resumed_history.validation_losses == expected_history.validation_losses
    assert resumed_history.validation_steps == expected_history.validation_steps
    assert resumed_history.tokens_seen == expected_history.tokens_seen
    assert resumed_history.optimizer_steps == expected_history.optimizer_steps == 2

    expected_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert expected_checkpoint["step"] == config.steps
    assert expected_checkpoint["optimizer_step"] == 2
    assert expected_checkpoint["history"]["optimizer_steps"] == 2

    resumed_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("optimizer", "scheduler", "generator_states", "torch_rng_state"):
        _assert_nested_equal(resumed_checkpoint[key], expected_checkpoint[key])
    assert (
        resumed_checkpoint["history"]["validation_losses"]
        == expected_checkpoint["history"]["validation_losses"]
    )
    assert (
        resumed_checkpoint["history"]["training_losses"]
        == expected_checkpoint["history"]["training_losses"]
    )
    assert (
        resumed_checkpoint["history"]["validation_steps"]
        == expected_checkpoint["history"]["validation_steps"]
    )
    assert (
        resumed_checkpoint["history"]["tokens_seen"]
        == expected_checkpoint["history"]["tokens_seen"]
    )


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


class _LinearLossLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets
        loss = self.weight * inputs.float().mean()
        return inputs.float(), loss


def test_final_incomplete_accumulation_window_uses_actual_microstep_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_ids = torch.arange(100, dtype=torch.long)
    config = TrainingConfig(
        steps=6,
        batch_size=2,
        block_size=1,
        learning_rate=0.01,
        eval_interval=10,
        eval_batches=1,
        seed=17,
        weight_decay=0.0,
        max_grad_norm=100.0,
        gradient_accumulation_steps=4,
    )
    captured_gradients: list[torch.Tensor] = []
    adamw = torch.optim.AdamW

    class CapturingAdamW(adamw):
        def step(self, closure: object = None) -> None:
            parameter = self.param_groups[0]["params"][0]
            captured_gradients.append(parameter.grad.detach().clone())
            super().step(closure=closure)

    monkeypatch.setattr(training_module.torch.optim, "AdamW", CapturingAdamW)
    train_language_model(
        _LinearLossLanguageModel(),
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=config,
    )

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    per_microstep: list[torch.Tensor] = []
    for _ in range(config.steps):
        inputs, _ = sample_batch(
            token_ids,
            batch_size=config.batch_size,
            block_size=config.block_size,
            generator=generator,
        )
        per_microstep.append(inputs.float().mean())
    torch.testing.assert_close(captured_gradients[0], sum(per_microstep[:4]) / 4)
    torch.testing.assert_close(captured_gradients[1], sum(per_microstep[4:]) / 2)


def test_learning_rate_schedule_uses_optimizer_update_budget() -> None:
    config = TrainingConfig(
        steps=10,
        batch_size=1,
        block_size=1,
        learning_rate=0.1,
        eval_interval=10,
        eval_batches=1,
        seed=1,
        warmup_steps=1,
        gradient_accumulation_steps=4,
    )

    assert [_lr_lambda(step, config) for step in range(4)] == [0.0, 1.0, 0.5, 0.0]


def test_warmup_steps_cannot_exceed_optimizer_update_budget() -> None:
    with pytest.raises(ValueError, match="update budget"):
        TrainingConfig(
            steps=3,
            batch_size=1,
            block_size=1,
            learning_rate=0.1,
            eval_interval=1,
            eval_batches=1,
            seed=1,
            warmup_steps=3,
            gradient_accumulation_steps=2,
        )
