import torch
from torch import nn

from transformer_lab.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
    load_adapter_state_dict,
    parameter_report,
)


def test_lora_linear_matches_its_frozen_base_before_adapter_training() -> None:
    base = nn.Linear(3, 2)
    adapter = LoRALinear(base, rank=2, alpha=4)
    inputs = torch.randn(4, 3)

    assert torch.equal(adapter(inputs), base(inputs))
    assert not base.weight.requires_grad
    assert all(parameter.requires_grad for parameter in adapter.adapter_parameters())


def test_lora_linear_adds_only_rank_factor_parameters() -> None:
    adapter = LoRALinear(nn.Linear(8, 6), rank=2, alpha=2)

    assert sum(parameter.numel() for parameter in adapter.adapter_parameters()) == 28


def test_inject_lora_replaces_only_named_linear_targets_and_reports_counts() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Module()
            self.attention.c_attn = nn.Linear(4, 6)
            self.other = nn.Linear(4, 4)

    model = Toy()
    report = inject_lora(model, LoRAConfig(rank=2, alpha=4, target_modules=("attention.c_attn",)))

    assert report.replaced_modules == ("attention.c_attn",)
    assert model.attention.c_attn.base.weight.requires_grad is False
    assert model.other.weight.requires_grad is False
    assert parameter_report(model).trainable == 20


def test_adapter_state_dict_can_be_loaded_without_changing_base_parameters() -> None:
    first = nn.Sequential(nn.Linear(4, 3))
    second = nn.Sequential(nn.Linear(4, 3))
    second.load_state_dict(first.state_dict())
    inject_lora(first, LoRAConfig(rank=2, alpha=2, target_modules=("0",)))
    inject_lora(second, LoRAConfig(rank=2, alpha=2, target_modules=("0",)))
    with torch.no_grad():
        first[0].a.fill_(0.5)
        first[0].b.fill_(0.25)

    state = adapter_state_dict(first)
    before = {
        name: value.clone() for name, value in second.state_dict().items() if ".base." in name
    }
    load_adapter_state_dict(second, state)

    assert state
    assert before
    assert torch.allclose(second[0].a, first[0].a)
    assert torch.equal(first(torch.ones(2, 4)), second(torch.ones(2, 4)))
