"""Runtime metadata helpers that avoid machine-specific secrets."""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(config_path: Path) -> dict[str, str | bool | None]:
    """Fingerprint the executable package and exact experiment configuration."""
    root = config_path.resolve().parent
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    files: list[Path] = []
    relative_root: Path | None = None
    if (root / "pyproject.toml").exists() and (root / "src").exists():
        relative_root = root
        files = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py"))]
    else:
        spec = importlib.util.find_spec("transformer_lab")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if locations:
            relative_root = Path(locations[0])
            files = sorted(relative_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(relative_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + path.read_bytes() + b"\0")
    commit, dirty = _git_state(root) if relative_root == root else (None, None)
    return {
        "source_tree_sha256": digest.hexdigest() if files else None,
        "source_tree_status": "available" if files else "unavailable",
        "config_sha256": file_sha256(config_path) if config_path.exists() else None,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    if not (root / ".git").exists():
        return None, None
    command = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    commit = subprocess.run(
        [*command, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        [*command, "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    if commit.returncode or status.returncode:
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Return package, Python, device, and memory metadata for a run."""
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": {
            package: _version(package)
            for package in ("numpy", "torch", "scikit-learn", "transformers", "joblib")
        },
    }
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "cuda": torch.version.cuda,
                "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
            }
        )
    else:
        metadata["cuda"] = None
    return metadata


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def peak_memory(device: torch.device) -> dict[str, int | None]:
    """Read peak CUDA allocation without failing on CPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
    return {
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
