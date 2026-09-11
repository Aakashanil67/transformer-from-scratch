import subprocess
import sys

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
