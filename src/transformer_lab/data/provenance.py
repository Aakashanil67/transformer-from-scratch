"""Write compact, deterministic records for dataset inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def _file_record(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_provenance(
    destination: Path,
    *,
    source: Mapping[str, str],
    files: Mapping[str, Path],
) -> None:
    """Record the declared source and checksums for the files used in a dataset."""
    manifest = {
        "source": dict(source),
        "files": {name: _file_record(path) for name, path in files.items()},
    }
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
