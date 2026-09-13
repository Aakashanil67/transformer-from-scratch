import pytest

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import (
    DecoderOnlyTransformer,
    attention_lora_parameter_count,
    transformer_parameter_count,
)


@pytest.mark.parametrize("bias", [True, False])
def test_transformer_parameter_formula_matches_instantiated_model(bias: bool) -> None:
    config = GPTConfig(
        vocab_size=31,
        block_size=11,
        n_layer=2,
        n_head=2,
        n_embd=8,
        bias=bias,
    )

    model = DecoderOnlyTransformer(config)

    assert transformer_parameter_count(config) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_attention_lora_parameter_formula_matches_the_declared_gpt2_adapter() -> None:
    config = GPTConfig(vocab_size=50_257, block_size=1_024, n_layer=12, n_head=12, n_embd=768)

    assert attention_lora_parameter_count(config, rank=4) == 221_184
