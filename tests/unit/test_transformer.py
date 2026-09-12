import pytest
import torch

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import DecoderOnlyTransformer


def build_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=16,
        block_size=4,
        n_layer=2,
        n_head=2,
        n_embd=8,
    )


def test_decoder_only_transformer_returns_logits_loss_and_tied_embeddings() -> None:
    model = DecoderOnlyTransformer(build_config())
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

    logits, loss = model(token_ids, targets)

    assert logits.shape == (2, 4, 16)
    assert loss is not None and torch.isfinite(loss)
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_layer_states_expose_each_parity_boundary() -> None:
    model = DecoderOnlyTransformer(build_config()).eval()
    token_ids = torch.tensor([[1, 2, 3]])

    states = model.layer_states(token_ids)

    assert len(states) == 3
    assert all(state.shape == (1, 3, 8) for state in states)
    assert torch.equal(model.hidden_states(token_ids), states[-1])
    assert torch.equal(states[1], model.blocks[0](states[0]))


def test_decoder_only_transformer_rejects_sequences_longer_than_its_context() -> None:
    model = DecoderOnlyTransformer(build_config())

    with pytest.raises(ValueError, match="block_size"):
        model(torch.tensor([[1, 2, 3, 4, 5]]))


def test_decoder_only_transformer_generate_crops_to_the_model_context() -> None:
    model = DecoderOnlyTransformer(build_config())
    prompt = torch.tensor([[1, 2, 3, 4]])
    torch.manual_seed(3)

    generated = model.generate(prompt, max_new_tokens=2)

    assert generated.shape == (1, 6)
    assert torch.equal(generated[:, :4], prompt)
    assert torch.all((generated >= 0) & (generated < 16))


@pytest.mark.parametrize(
    "token_ids",
    [torch.tensor([1, 2]), torch.tensor([[1.0, 2.0]]), torch.empty((1, 0), dtype=torch.long)],
)
def test_decoder_only_transformer_rejects_invalid_token_inputs(token_ids: torch.Tensor) -> None:
    model = DecoderOnlyTransformer(build_config())

    with pytest.raises(ValueError, match="token_ids"):
        model(token_ids)


def test_decoder_only_transformer_rejects_out_of_range_targets() -> None:
    model = DecoderOnlyTransformer(build_config())

    with pytest.raises(ValueError, match="target shape"):
        model(torch.tensor([[1, 2]]), torch.tensor([[1]]))


def test_decoder_only_transformer_greedy_generation_is_deterministic() -> None:
    model = DecoderOnlyTransformer(build_config()).eval()
    prompt = torch.tensor([[1, 2]])

    first = model.generate(prompt, max_new_tokens=3, do_sample=False)
    second = model.generate(prompt, max_new_tokens=3, do_sample=False)

    assert torch.equal(first, second)


def test_new_transformer_uses_scaled_gpt_initialisation() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(
        GPTConfig(vocab_size=256, block_size=8, n_layer=4, n_head=4, n_embd=64)
    )

    embedding_std = model.token_embedding.weight.std().item()
    residual_std = model.blocks[0].attention.c_proj.weight.std().item()

    assert 0.018 < embedding_std < 0.022
    assert 0.006 < residual_std < 0.008
