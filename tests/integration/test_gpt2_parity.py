import os

import pytest
import torch

pytestmark = pytest.mark.integration


def test_pinned_gpt2_parity_is_opt_in() -> None:
    """Keep the network-backed parity check out of ordinary test runs."""
    if os.environ.get("RUN_NETWORK_EXPERIMENTS") != "1":
        pytest.skip("set RUN_NETWORK_EXPERIMENTS=1 to run the pinned checkpoint check")
    from pathlib import Path

    from transformer_lab.config_io import load_experiment_config
    from transformer_lab.experiments.parity import run

    root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run(load_experiment_config(root / "configs/gpt2/parity.toml"), device=device)
    assert '"status": "completed"' in result.read_text(encoding="utf-8")
