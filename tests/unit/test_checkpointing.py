from pathlib import Path

import torch
from torch import nn

from transformer_lab.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    inputs = torch.ones(1, 2)
    (model(inputs).sum()).backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    destination = tmp_path / "checkpoint.pt"

    save_checkpoint(destination, model=model, optimizer=optimizer, step=3)
    restored = nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    payload = load_checkpoint(destination, model=restored, optimizer=restored_optimizer)

    assert payload["step"] == 3
    assert all(torch.equal(expected[name], value) for name, value in restored.state_dict().items())
