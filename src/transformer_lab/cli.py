"""Command-line entry point for the transformer experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from transformer_lab import __version__
from transformer_lab.commands import (
    evaluate_sentiment,
    sample,
    train_lm,
    train_sentiment,
    verify_gpt2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformer-lab", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-lm", "sample", "verify-gpt2", "train-sentiment", "evaluate-sentiment"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name in {"train-lm", "sample", "train-sentiment"}:
            command.add_argument("--seed", type=int)
        if name in {"train-lm", "train-sentiment"}:
            command.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        if args.command == "train-lm":
            result = train_lm(
                Path(args.config), device_name=args.device, seed=args.seed, resume=args.resume
            )
            print(f"Wrote {result}")
        elif args.command == "sample":
            print(sample(Path(args.config), device_name=args.device, seed=args.seed))
        elif args.command == "verify-gpt2":
            result = verify_gpt2(Path(args.config), device_name=args.device)
            print(f"Wrote {result}")
        elif args.command == "train-sentiment":
            result = train_sentiment(
                Path(args.config),
                device_name=args.device,
                seed=args.seed,
                resume=args.resume,
            )
            print(f"Wrote {result}")
        else:
            result = evaluate_sentiment(Path(args.config), device_name=args.device)
            print(f"Verified {result}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
