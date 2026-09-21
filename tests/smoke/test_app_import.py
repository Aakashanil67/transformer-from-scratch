def test_streamlit_app_imports() -> None:
    import app.playground  # noqa: F401


def test_app_ignores_metrics_from_non_completed_records() -> None:
    from app.playground import _completed_test_metrics

    assert _completed_test_metrics({"status": "failed", "metrics": None}) is None
    assert _completed_test_metrics({"status": "unavailable", "metrics": None}) is None
    assert _completed_test_metrics({"status": "completed", "metrics": {}}) is None


def test_app_returns_test_metrics_from_completed_record() -> None:
    from app.playground import _completed_test_metrics

    metrics = {"macro_f1": 0.8, "accuracy": 0.9}

    assert _completed_test_metrics({"status": "completed", "metrics": {"test": metrics}}) == metrics


def test_app_ignores_malformed_completed_test_metrics() -> None:
    from app.playground import _completed_test_metrics

    assert _completed_test_metrics({"status": "completed", "metrics": {"test": {}}}) is None
    assert (
        _completed_test_metrics({"status": "completed", "metrics": {"test": {"macro_f1": "0.8"}}})
        is None
    )


def test_app_ignores_out_of_range_aggregate_metrics() -> None:
    from app.playground import _comparison_summary

    record = {
        "status": "completed",
        "metrics": {
            "methods": {
                "lora": {
                    "seeds": [17, 23, 41],
                    "macro_f1": {"mean": 1.2, "standard_deviation": 0.01},
                    "accuracy": {"mean": 0.9, "standard_deviation": -0.01},
                }
            }
        },
    }

    assert _comparison_summary(record, "lora") is None


def test_app_uses_a_hash_verified_artifact_only(tmp_path) -> None:
    import hashlib
    import json

    from app.playground import _verified_artifact

    model = tmp_path / "model.pt"
    model.write_bytes(b"original")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "completed",
                "metrics": {},
                "artifacts": {"model": {"sha256": digest}},
            }
        ),
        encoding="utf-8",
    )

    verified, error = _verified_artifact(result, (tmp_path,), "model")
    assert verified == model
    assert error is None

    model.write_bytes(b"changed")
    verified, error = _verified_artifact(result, (tmp_path,), "model")
    assert verified is None
    assert error is not None and "hash" in error
