import pytest
import torch

from transformer_lab.models.bigram import BigramLanguageModel
from transformer_lab.training import TrainingConfig, train_language_model

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_model_can_train_from_cpu_corpora() -> None:
    token_ids = torch.tensor([0, 1, 2, 3] * 40)
    model = BigramLanguageModel(vocab_size=4).cuda()

    history = train_language_model(
        model,
        train_token_ids=token_ids,
        validation_token_ids=token_ids,
        config=TrainingConfig(
            steps=3,
            batch_size=2,
            block_size=2,
            learning_rate=0.05,
            eval_interval=2,
            eval_batches=1,
            seed=17,
            gradient_accumulation_steps=2,
        ),
    )

    assert history.optimizer_steps == 2
    assert history.tokens_seen == 12
