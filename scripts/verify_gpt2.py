"""Run the pinned GPT-2 parity check from a source checkout."""

from __future__ import annotations

from pathlib import Path

from transformer_lab.config_io import load_experiment_config
from transformer_lab.experiments.parity import run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run(load_experiment_config(root / "configs" / "gpt2" / "parity.toml"))


if __name__ == "__main__":
    main()
