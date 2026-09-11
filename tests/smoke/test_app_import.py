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
