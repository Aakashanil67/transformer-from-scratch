"""Verify base, app and all wheel install surfaces in isolated environments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path, log: list[dict[str, Any]]) -> None:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    log.append(
        {
            "command": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def _venv_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _verify_surface(
    environment: Path,
    wheel: Path,
    extra: str | None,
    *,
    log: list[dict[str, Any]],
) -> None:
    python = _venv_python(environment)
    venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    requirement = str(wheel) + (f"[{extra}]" if extra else "")
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-c",
            str(ROOT / "requirements/constraints-cpu.txt"),
            requirement,
        ],
        cwd=Path(tempfile.gettempdir()),
        log=log,
    )
    check = (
        "import pathlib, sys, transformer_lab; "
        "path = pathlib.Path(transformer_lab.__file__).resolve(); "
        "assert str(path).startswith(str(pathlib.Path(sys.prefix).resolve())), path; "
        "print(path)"
    )
    _run([str(python), "-c", check], cwd=Path(tempfile.gettempdir()), log=log)
    _run(
        [str(python), "-m", "transformer_lab.cli", "--help"],
        cwd=Path(tempfile.gettempdir()),
        log=log,
    )
    if extra in {"app", "all"}:
        _run(
            [
                str(python),
                "-c",
                "from transformer_lab.inference import generation, sentiment; "
                "print('inference imports ok')",
            ],
            cwd=Path(tempfile.gettempdir()),
            log=log,
        )
    if extra in {None, "all"}:
        fixture = environment.parent / f"{environment.name}-fixture"
        fixture.mkdir()
        (fixture / "train.txt").write_bytes(b"ab" * 100)
        (fixture / "validation.txt").write_bytes(b"ab" * 100)
        (fixture / "config.toml").write_text(
            """
[experiment]
name = "installed-fixture"
kind = "lm"
[model]
model_type = "bigram"
vocab_size = 256
[data]
train = "train.txt"
validation = "validation.txt"
[optimization]
steps = 2
batch_size = 2
block_size = 1
learning_rate = 0.1
eval_interval = 1
eval_batches = 1
[output]
checkpoint = "model.pt"
result = "result.json"
[generation]
prompt = "a"
max_new_tokens = 2
""",
            encoding="utf-8",
        )
        _run(
            [
                str(python),
                "-m",
                "transformer_lab.cli",
                "train-lm",
                "--config",
                str(fixture / "config.toml"),
                "--device",
                "cpu",
            ],
            cwd=fixture,
            log=log,
        )
        _run(
            [
                str(python),
                "-m",
                "transformer_lab.cli",
                "sample",
                "--config",
                str(fixture / "config.toml"),
                "--device",
                "cpu",
                "--seed",
                "17",
            ],
            cwd=fixture,
            log=log,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "artifacts/verification/install.json")
    parser.add_argument("--wheel", type=Path, help="use an existing wheel instead of building one")
    args = parser.parse_args(argv)
    log: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "surfaces": {},
    }
    try:
        with tempfile.TemporaryDirectory(prefix="transformer-lab-install-") as temporary:
            temporary_path = Path(temporary)
            wheel = args.wheel.resolve() if args.wheel else temporary_path / "dist"
            if args.wheel is None:
                _run(
                    [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel)],
                    cwd=ROOT,
                    log=log,
                )
                wheels = sorted(wheel.glob("*.whl"))
                if len(wheels) != 1:
                    raise RuntimeError(f"expected one built wheel, found {len(wheels)}")
                wheel = wheels[0]
            if not wheel.is_file():
                raise FileNotFoundError(f"wheel does not exist: {wheel}")
            for name, extra in (("base", None), ("app", "app"), ("all", "all")):
                environment = temporary_path / name
                _verify_surface(environment, wheel, extra, log=log)
                result["surfaces"][name] = "passed"
    except (OSError, RuntimeError) as error:
        result["status"] = "failed"
        result["error"] = str(error)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"result": result, "commands": log}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"error: {error}", file=sys.stderr)
        return 2
    result["status"] = "passed"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(
        json.dumps({"result": result, "commands": log}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
