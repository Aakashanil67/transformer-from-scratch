import json

import pytest

from transformer_lab.evaluation.reporting import (
    format_gib,
    render_class_metrics_markdown,
    render_comparison_markdown,
    validate_result_payload,
)


def _comparison() -> dict:
    return {
        "status": "completed",
        "metrics": {
            "methods": {
                "tfidf": {
                    "macro_f1": {"mean": 0.800689, "standard_deviation": None},
                    "accuracy": {"mean": 0.853282, "standard_deviation": None},
                    "seeds": [17],
                },
                "full": {
                    "macro_f1": {"mean": 0.868692, "standard_deviation": 0.003672},
                    "accuracy": {"mean": 0.899614, "standard_deviation": 0.003861},
                    "seeds": [17, 23, 41],
                },
            }
        },
    }


def test_format_gib_uses_binary_units() -> None:
    assert format_gib(1_029_140_480) == "0.958 GiB"


def test_rendered_table_has_matching_columns_and_labels_single_seed() -> None:
    table = render_comparison_markdown(_comparison())
    lines = table.splitlines()

    assert "Peak allocated memory (GiB)" in lines[2]
    assert "single seed" in table
    assert all(line.count("|") == 6 for line in lines[2:5])


def test_validate_result_payload_rejects_nonfinite_values() -> None:
    payload = _comparison()
    payload["metrics"]["methods"]["full"]["macro_f1"]["mean"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_result_payload(payload)


def test_rendered_json_is_strictly_serialisable() -> None:
    payload = _comparison()
    validate_result_payload(json.loads(json.dumps(payload)))


def test_reporting_renders_class_support_recall_and_f1() -> None:
    rendered = render_class_metrics_markdown(
        {
            "metrics": {
                "test": {
                    "per_class": {
                        "negative": {"support": 2, "recall": 0.5, "f1": 0.4},
                        "positive": {"support": 3, "recall": 0.75, "f1": 0.7},
                    }
                }
            }
        }
    )

    assert "| negative | 2 | 0.5000 | 0.4000 |" in rendered
    assert rendered.count("|") == 20


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"metrics": None}, "no metrics"),
        ({"metrics": {"methods": {}}}, "no method"),
        (
            {
                "metrics": {
                    "methods": {
                        "x": {
                            "seeds": [],
                            "macro_f1": {"mean": 0.5, "standard_deviation": None},
                            "accuracy": {"mean": 0.5, "standard_deviation": None},
                        }
                    }
                }
            },
            "integer seeds",
        ),
    ],
)
def test_reporting_rejects_incomplete_summaries(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_result_payload(payload)


def test_format_gib_rejects_invalid_memory() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_gib(-1)
    with pytest.raises(ValueError, match="numeric"):
        format_gib("1")


def test_class_metric_renderer_rejects_missing_sections() -> None:
    with pytest.raises(ValueError, match="per-class"):
        render_class_metrics_markdown({"metrics": {"test": {}}})
