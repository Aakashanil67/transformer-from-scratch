import torch

from transformer_lab.config import GPTConfig
from transformer_lab.gpt2 import gpt2_small_config, load_huggingface_gpt2_weights
from transformer_lab.models.transformer import DecoderOnlyTransformer


def test_gpt2_small_config_matches_the_124m_architecture() -> None:
    config = gpt2_small_config()

    assert (config.vocab_size, config.block_size) == (50_257, 1_024)
    assert (config.n_layer, config.n_head, config.n_embd) == (12, 12, 768)
    assert config.dropout == 0.0


def test_loader_maps_gpt2_attention_projection_names_without_rewriting_c_attn() -> None:
    config = GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=4)
    source_model = DecoderOnlyTransformer(config)
    target_model = DecoderOnlyTransformer(config)
    state = source_model.state_dict()
    reference = {
        "transformer.wte.weight": state["token_embedding.weight"],
        "transformer.wpe.weight": state["position_embedding.weight"],
        "transformer.ln_f.weight": state["ln_f.weight"],
        "transformer.ln_f.bias": state["ln_f.bias"],
        "transformer.h.0.ln_1.weight": state["blocks.0.ln_1.weight"],
        "transformer.h.0.ln_1.bias": state["blocks.0.ln_1.bias"],
        "transformer.h.0.ln_2.weight": state["blocks.0.ln_2.weight"],
        "transformer.h.0.ln_2.bias": state["blocks.0.ln_2.bias"],
        "transformer.h.0.attn.c_attn.weight": state["blocks.0.attention.c_attn.weight"].t(),
        "transformer.h.0.attn.c_attn.bias": state["blocks.0.attention.c_attn.bias"],
        "transformer.h.0.attn.c_proj.weight": state["blocks.0.attention.c_proj.weight"].t(),
        "transformer.h.0.attn.c_proj.bias": state["blocks.0.attention.c_proj.bias"],
        "transformer.h.0.mlp.c_fc.weight": state["blocks.0.mlp.c_fc.weight"].t(),
        "transformer.h.0.mlp.c_fc.bias": state["blocks.0.mlp.c_fc.bias"],
        "transformer.h.0.mlp.c_proj.weight": state["blocks.0.mlp.c_proj.weight"].t(),
        "transformer.h.0.mlp.c_proj.bias": state["blocks.0.mlp.c_proj.bias"],
    }

    load_huggingface_gpt2_weights(target_model, reference)

    assert torch.equal(
        target_model.state_dict()["blocks.0.attention.c_attn.weight"],
        state["blocks.0.attention.c_attn.weight"],
    )


def test_loader_reports_missing_reference_keys() -> None:
    config = GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=4)
    model = DecoderOnlyTransformer(config)

    try:
        load_huggingface_gpt2_weights(model, {})
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing reference keys were accepted")
