"""Runtime metadata helpers that avoid machine-specific secrets."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch


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
