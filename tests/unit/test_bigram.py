import math

import torch

from transformer_lab.models.bigram import BigramLanguageModel


def test_bigram_forward_returns_vocab_logits_and_cross_entropy_loss() -> None:
    model = BigramLanguageModel(vocab_size=5)
    with torch.no_grad():
        model.token_logits.weight.zero_()
    token_ids = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])

    logits, loss = model(token_ids, targets)

    assert logits.shape == (2, 3, 5)
    assert loss is not None
    assert torch.isclose(loss, torch.tensor(math.log(5.0)), atol=1e-6)


def test_bigram_generate_appends_requested_number_of_tokens() -> None:
    model = BigramLanguageModel(vocab_size=4)
    prompt = torch.tensor([[1, 2]])
    torch.manual_seed(7)

    generated = model.generate(prompt, max_new_tokens=3)

    assert generated.shape == (1, 5)
    assert torch.equal(generated[:, :2], prompt)
    assert torch.all((generated >= 0) & (generated < 4))
