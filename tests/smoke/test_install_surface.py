import subprocess
import sys
import tomllib
from pathlib import Path

import transformer_lab


def test_package_exposes_a_release_version() -> None:
    assert transformer_lab.__version__ != "0.0.0"


def test_installed_module_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "transformer_lab.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "train-lm" in result.stdout


def test_app_extra_declares_the_inference_dependencies() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    app_dependencies = set(pyproject["project"]["optional-dependencies"]["app"])

    assert any(dependency.startswith("scikit-learn") for dependency in app_dependencies)
    assert any(dependency.startswith("transformers") for dependency in app_dependencies)


def test_install_verifier_exposes_a_non_mutating_help_contract() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_install.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "isolated environments" in result.stdout


def test_release_verifier_exposes_explicit_optional_modes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_release.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--include-install" in result.stdout
    assert "--include-gpu" in result.stdout
