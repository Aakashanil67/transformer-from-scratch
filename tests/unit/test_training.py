from pathlib import Path

import pytest
import torch

from transformer_lab.models.bigram import BigramLanguageModel
from transformer_lab.training import TrainingConfig, train_language_model


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
