"""Small, stable result records for reproducible experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    """A serialisable summary of one completed or failed run."""

    run_id: str
    status: str
    model: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] | None = field(default_factory=dict)
    error: dict[str, Any] | None = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.schema_version not in {2, 3}:
            raise ValueError("schema_version must be 2 or 3")
        if self.status not in {"completed", "failed", "unavailable"}:
            raise ValueError("status must be completed, failed, or unavailable")
        if self.status == "completed" and self.metrics is None:
            raise ValueError("completed records require metrics")
        if self.status != "completed" and self.metrics is not None:
            raise ValueError("failed and unavailable records require metrics = None")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible mapping."""
        payload = {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}
        return json.loads(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )

    def write(self, path: Path) -> Path:
        """Write this record atomically enough for a local experiment."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        return path


def failed_record(run_id: str, error: BaseException) -> ExperimentRecord:
    """Build a truthful failed-run record without capturing local secrets."""
    return ExperimentRecord(
        run_id=run_id,
        status="failed",
        metrics=None,
        error={"type": type(error).__name__, "message": _public_error(str(error))},
    )


def _public_error(message: str) -> str:
    return re.sub(r"(?:[A-Za-z]:\\|/)[^\s\"']+", "<path>", message)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value
