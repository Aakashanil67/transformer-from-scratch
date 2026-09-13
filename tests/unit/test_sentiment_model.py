import pytest
import torch

from transformer_lab.config import GPTConfig
from transformer_lab.models.sentiment import SentimentClassifier


def test_sentiment_classifier_pools_the_last_unmasked_token() -> None:
    model = SentimentClassifier(
        GPTConfig(vocab_size=20, block_size=6, n_layer=1, n_head=2, n_embd=8), num_labels=3
    ).eval()
    inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    logits, loss = model(inputs, attention_mask=mask, labels=torch.tensor([1, 2]))

    assert logits.shape == (2, 3)
    assert loss is not None and torch.isfinite(loss)


def test_sentiment_classifier_rejects_an_invalid_attention_mask() -> None:
    model = SentimentClassifier(
        GPTConfig(vocab_size=20, block_size=6, n_layer=1, n_head=2, n_embd=8), num_labels=3
    )

    try:
        model(torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1]]))
    except ValueError as error:
        assert "attention_mask" in str(error)
    else:
        raise AssertionError("invalid attention mask was accepted")


@pytest.mark.parametrize(
    "mask",
    [torch.tensor([[1, 0, 1]]), torch.tensor([[0, 1, 1]]), torch.tensor([[1, 2, 0]])],
)
def test_sentiment_classifier_requires_a_right_padded_binary_mask(mask: torch.Tensor) -> None:
    model = SentimentClassifier(
        GPTConfig(vocab_size=20, block_size=6, n_layer=1, n_head=2, n_embd=8), num_labels=3
    )

    with pytest.raises(ValueError, match="right-padded binary"):
        model(torch.tensor([[1, 2, 3]]), attention_mask=mask)
