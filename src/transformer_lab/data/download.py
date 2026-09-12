"""Download raw datasets without adding a runtime dependency."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen


def download_file(url: str, destination: Path, *, expected_sha256: str | None = None) -> Path:
    """Download ``url`` to ``destination`` and return the destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_sha256 is not None:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_sha256.lower():
            return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if expected_sha256 is not None:
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != expected_sha256.lower():
                raise ValueError(
                    f"download SHA-256 mismatch: expected {expected_sha256}, received {digest}"
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
