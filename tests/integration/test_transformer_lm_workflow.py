import pytest
import torch

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import DecoderOnlyTransformer
from transformer_lab.training import TrainingConfig, train_language_model

pytestmark = pytest.mark.integration


def test_scratch_transformer_learns_a_repeating_byte_corpus() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(
        GPTConfig(vocab_size=4, block_size=4, n_layer=1, n_head=2, n_embd=8)
    )
    token_ids = torch.tensor([0, 1, 2, 3] * 100)

    history = train_language_model(
        model,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=TrainingConfig(
            steps=12,
            batch_size=4,
            block_size=4,
            learning_rate=0.02,
            eval_interval=6,
            eval_batches=4,
            seed=17,
            gradient_accumulation_steps=2,
        ),
    )

    assert history.validation_losses[-1] < history.validation_losses[0]
    assert history.optimizer_steps == 6
