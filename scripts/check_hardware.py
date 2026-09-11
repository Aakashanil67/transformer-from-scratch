"""Print the compute device available to the experiment runner."""

from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    result: dict[str, object] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "device": properties.name,
                "memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    else:
        result["device"] = "cpu"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
