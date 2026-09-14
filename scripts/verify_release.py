"""Run and record the offline release verification gate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _command_result(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "artifacts/verification/release.json")
    parser.add_argument("--include-install", action="store_true")
    parser.add_argument("--include-gpu", action="store_true")
    args = parser.parse_args(argv)
    python = str(Path(sys.executable).resolve())
    checks: list[dict[str, Any]] = []
    commands = [
        [python, "-m", "ruff", "format", "--check", "src", "tests", "scripts", "app"],
        [python, "-m", "ruff", "check", "src", "tests", "scripts", "app"],
        [python, "-m", "pytest", "-q"],
        [
            python,
            "-m",
            "pytest",
            "--cov=transformer_lab",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
        ],
        [python, "-m", "build"],
        [python, "-m", "pip", "check"],
        [
            python,
            "scripts/render_results.py",
            "--results",
            "reports/results",
            "--output",
            "reports/generated-results.md",
            "--check",
        ],
        [
            python,
            "scripts/render_results.py",
            "--results",
            "reports/results/v3",
            "--output",
            "reports/generated-results-v3.md",
            "--check",
        ],
    ]
    if args.include_install:
        commands.append([python, "scripts/verify_install.py"])
    else:
        checks.append(
            {
                "status": "skipped",
                "name": "isolated_install_surfaces",
                "reason": "use --include-install; installs dependencies in temporary environments",
            }
        )
    for command in commands:
        checks.append(_command_result(command, cwd=ROOT))
    if args.include_gpu:
        gpu_python = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
        if gpu_python.exists():
            checks.append(
                _command_result(
                    [str(gpu_python), "-m", "pytest", "-q", "-m", "integration"], cwd=ROOT
                )
            )
        else:
            checks.append(
                {
                    "status": "skipped",
                    "name": "cuda_integration",
                    "reason": f"missing GPU environment: {gpu_python}",
                }
            )
    else:
        checks.append(
            {
                "status": "skipped",
                "name": "cuda_integration",
                "reason": "use --include-gpu; CUDA and model downloads are opt-in",
            }
        )
    result = {
        "status": "passed" if all(check["status"] != "failed" for check in checks) else "failed",
        "platform": platform.platform(),
        "python": sys.version,
        "checks": checks,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks)}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
