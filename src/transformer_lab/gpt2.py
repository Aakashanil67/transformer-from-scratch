"""GPT-2 small architecture settings and state-dict conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from transformer_lab.config import GPTConfig
from transformer_lab.models.transformer import DecoderOnlyTransformer

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def gpt2_small_config() -> GPTConfig:
    """Return the 124M GPT-2 architecture without loading any weights."""
    return GPTConfig(
        vocab_size=50_257,
        block_size=1_024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )


@dataclass(frozen=True)
class WeightLoadReport:
    """Summary of a GPT-2 state-dict conversion."""

    loaded: int
    expected: int
    ignored_reference_keys: tuple[str, ...]


def load_huggingface_gpt2_weights(
    model: DecoderOnlyTransformer,
    reference_state: dict[str, Tensor],
) -> WeightLoadReport:
    """Copy a Hugging Face GPT-2 state dict into this repository's module layout."""
    target_state = model.state_dict()
    direct = {
        "transformer.wte.weight": "token_embedding.weight",
        "transformer.wpe.weight": "position_embedding.weight",
        "transformer.ln_f.weight": "ln_f.weight",
        "transformer.ln_f.bias": "ln_f.bias",
    }
    transposed_suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    for index in range(model.config.n_layer):
        source_prefix = f"transformer.h.{index}."
        target_prefix = f"blocks.{index}."
        direct.update(
            {
                f"{source_prefix}ln_1.weight": f"{target_prefix}ln_1.weight",
                f"{source_prefix}ln_1.bias": f"{target_prefix}ln_1.bias",
                f"{source_prefix}ln_2.weight": f"{target_prefix}ln_2.weight",
                f"{source_prefix}ln_2.bias": f"{target_prefix}ln_2.bias",
                f"{source_prefix}attn.c_attn.bias": f"{target_prefix}attention.c_attn.bias",
                f"{source_prefix}attn.c_proj.bias": f"{target_prefix}attention.c_proj.bias",
                f"{source_prefix}mlp.c_fc.bias": f"{target_prefix}mlp.c_fc.bias",
                f"{source_prefix}mlp.c_proj.bias": f"{target_prefix}mlp.c_proj.bias",
            }
        )
        for suffix in transposed_suffixes:
            target_suffix = (
                f"attention.{suffix.removeprefix('attn.')}"
                if suffix.startswith("attn.")
                else suffix
            )
            direct[f"{source_prefix}{suffix}"] = f"{target_prefix}{target_suffix}"

    missing = sorted(set(direct) - set(reference_state))
    if missing:
        raise ValueError(f"missing reference keys: {missing[:5]}")
    missing_targets = sorted(set(direct.values()) - {"lm_head.weight"} - set(target_state))
    if missing_targets:
        raise ValueError(f"missing target keys: {missing_targets[:5]}")

    with torch.no_grad():
        for source_name, target_name in direct.items():
            source = reference_state[source_name]
            target = target_state[target_name]
            if source_name.endswith(transposed_suffixes):
                source = source.t()
            if source.shape != target.shape:
                raise ValueError(
                    f"shape mismatch for {source_name}: {source.shape} != {target.shape}"
                )
            target.copy_(source)
    return WeightLoadReport(
        loaded=len(direct),
        expected=len(direct),
        ignored_reference_keys=tuple(sorted(set(reference_state) - set(direct))),
    )
