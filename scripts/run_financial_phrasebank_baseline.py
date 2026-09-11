"""Run the Financial PhraseBank reference model from a source checkout."""

from __future__ import annotations

from pathlib import Path

from transformer_lab.config_io import load_experiment_config
from transformer_lab.experiments.baseline import run_baseline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_baseline(load_experiment_config(root / "configs" / "sentiment" / "baseline.toml"))


if __name__ == "__main__":
    main()
