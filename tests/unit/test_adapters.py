import torch
from torch import nn

from transformer_lab.adapters import LoRAConfig, inject_lora, load_adapter, save_adapter


def test_adapter_artifact_restores_identical_logits(tmp_path) -> None:
    first = nn.Sequential(nn.Linear(4, 3))
    second = nn.Sequential(nn.Linear(4, 3))
    second.load_state_dict(first.state_dict())
    config = LoRAConfig(rank=2, alpha=2, target_modules=("0",))
    inject_lora(first, config)
    inject_lora(second, config)
    with torch.no_grad():
        first[0].a.fill_(0.25)
        first[0].b.fill_(0.5)
    path = save_adapter(tmp_path / "adapter.pt", first, config)

    metadata = load_adapter(path, second)

    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert metadata["rank"] == 2
    assert torch.equal(first(inputs), second(inputs))
