import json
from types import MappingProxyType

import pytest

from transformer_lab.experiments.records import ExperimentRecord, failed_record


def test_experiment_record_round_trips_with_a_stable_schema(tmp_path) -> None:
    record = ExperimentRecord(run_id="demo", status="completed", metrics={"macro_f1": 0.8})
    destination = record.write(tmp_path / "result.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["run_id"] == "demo"
    assert payload["metrics"] == {"macro_f1": 0.8}
    assert payload["timestamp_utc"].endswith("Z")


def test_experiment_record_keeps_source_and_artifact_fingerprints() -> None:
    record = ExperimentRecord(
        run_id="fingerprinted",
        status="completed",
        provenance={"source_tree_sha256": "a" * 64},
        artifacts={"model": {"sha256": "b" * 64}},
        metrics={},
    )

    payload = record.to_dict()

    assert payload["provenance"]["source_tree_sha256"] == "a" * 64
    assert payload["artifacts"]["model"]["sha256"] == "b" * 64


def test_experiment_record_serialises_immutable_nested_mappings() -> None:
    record = ExperimentRecord(
        run_id="immutable",
        status="completed",
        optimization=MappingProxyType({"batch_size": 1}),
        metrics={},
    )

    assert record.to_dict()["optimization"] == {"batch_size": 1}


def test_failed_record_keeps_metrics_null_and_records_the_error() -> None:
    record = failed_record("run", MemoryError("out of memory"))

    assert record.status == "failed"
    assert record.metrics is None
    assert record.error == {"type": "MemoryError", "message": "out of memory"}


def test_record_rejects_an_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        ExperimentRecord(run_id="demo", status="finished", metrics={})


def test_completed_record_requires_metrics() -> None:
    with pytest.raises(ValueError, match="metrics"):
        ExperimentRecord(run_id="demo", status="completed", metrics=None)


def test_failed_record_removes_absolute_paths_from_public_message() -> None:
    record = failed_record("run", RuntimeError(r"C:\Users\aakas\private\model.pt failed"))

    assert "aakas" not in record.error["message"]
    assert "<path>" in record.error["message"]
