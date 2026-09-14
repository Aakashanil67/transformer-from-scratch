import json

import pytest

from transformer_lab.experiments.selection import (
    load_selection_manifest,
    load_selection_protocol,
    select_candidates,
    write_selection_manifest,
)


def _candidate(candidate_id: str, method: str, macro_f1: float, loss: float, order: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "method": method,
        "order": order,
        "status": "completed",
        "validation": {"macro_f1": macro_f1, "loss": loss},
        "config_digest": f"digest-{candidate_id}",
        "result": f"{candidate_id}.json",
    }


def test_selection_ranks_validation_only_with_deterministic_ties() -> None:
    selected = select_candidates(
        [
            _candidate("head-1", "head_only", 0.70, 0.40, 0),
            _candidate("head-2", "head_only", 0.70, 0.30, 1),
            _candidate("head-3", "head_only", 0.70, 0.30, 2),
            _candidate("lora-1", "lora", 0.80, 0.50, 0),
        ]
    )

    assert selected["head_only"]["candidate_id"] == "head-2"
    assert selected["lora"]["candidate_id"] == "lora-1"
    assert selected["head_only"]["selection_metric"] == {
        "validation_macro_f1": 0.7,
        "validation_loss": 0.3,
    }


def test_selection_preserves_failures_but_does_not_select_them() -> None:
    failed = _candidate("head-failed", "head_only", 0.0, 0.0, 0)
    failed["status"] = "failed"
    failed["error"] = {"type": "OutOfMemoryError", "message": "fixture"}

    selected = select_candidates([failed, _candidate("head-ok", "head_only", 0.2, 1.0, 1)])

    assert selected["head_only"]["candidate_id"] == "head-ok"
    assert selected["head_only"]["considered_candidates"] == ["head-failed", "head-ok"]


def test_selection_rejects_test_metrics_and_test_artifacts() -> None:
    candidate = _candidate("head-1", "head_only", 0.7, 0.4, 0)
    candidate["test"] = {"macro_f1": 0.9}

    with pytest.raises(ValueError, match="test"):
        select_candidates([candidate])


def test_selection_manifest_round_trip_and_schema_checks(tmp_path) -> None:
    candidates = [_candidate("head-1", "head_only", 0.7, 0.4, 0)]
    path = write_selection_manifest(
        tmp_path / "selection.json",
        protocol={"name": "fixture", "split_seed": 17},
        candidates=candidates,
    )

    loaded = load_selection_manifest(path)

    assert loaded["schema_version"] == 1
    assert loaded["selection"]["head_only"]["candidate_id"] == "head-1"
    assert loaded["protocol"]["split_seed"] == 17

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["head_only"]["test"] = {"macro_f1": 0.9}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        load_selection_manifest(path)


def test_selection_rejects_empty_and_malformed_candidates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_candidates([])
    base = _candidate("head-1", "head_only", 0.7, 0.4, 0)
    malformed = [
        {**base, "candidate_id": ""},
        {**base, "candidate_id": "head-2", "method": ""},
        {**base, "candidate_id": "head-2", "order": -1},
        {**base, "candidate_id": "head-2", "status": "running"},
        {**base, "candidate_id": "head-2", "validation": None},
        {
            **base,
            "candidate_id": "head-2",
            "validation": {"macro_f1": float("nan"), "loss": 0.2},
        },
    ]
    messages = [
        "candidate_id",
        "no method",
        "invalid order",
        "invalid status",
        "no validation",
        "invalid",
    ]
    for candidate, message in zip(malformed, messages, strict=True):
        with pytest.raises(ValueError, match=message):
            select_candidates([candidate])

    duplicate = _candidate("head-1", "head_only", 0.7, 0.4, 0)
    with pytest.raises(ValueError, match="duplicate"):
        select_candidates([duplicate, dict(duplicate)])


def test_selection_records_method_without_a_completed_candidate() -> None:
    candidate = _candidate("head-unavailable", "head_only", 0.0, 0.0, 0)
    candidate["status"] = "unavailable"

    selected = select_candidates([candidate])

    assert selected["head_only"]["status"] == "unavailable"
    assert selected["head_only"]["candidate_id"] is None


def test_selection_protocol_reads_inline_validation_summaries(tmp_path) -> None:
    path = tmp_path / "protocol.toml"
    path.write_text(
        """
[protocol]
name = "fixture"
split_seed = 17

[[candidates]]
candidate_id = "head-1"
method = "head_only"
order = 0
validation_macro_f1 = 0.7
validation_loss = 0.4
""",
        encoding="utf-8",
    )

    protocol, candidates = load_selection_protocol(path)

    assert protocol["name"] == "fixture"
    assert candidates[0]["validation"] == {"macro_f1": 0.7, "loss": 0.4}


def test_selection_protocol_reads_relative_summary_and_preserves_failure(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"validation": {"macro_f1": 0.8, "loss": 0.3}, "status": "completed"}),
        encoding="utf-8",
    )
    path = tmp_path / "protocol.toml"
    path.write_text(
        """
[protocol]
name = "fixture"

[[candidates]]
candidate_id = "head-1"
method = "head_only"
order = 0
summary = "summary.json"

[[candidates]]
candidate_id = "head-2"
method = "head_only"
order = 1
status = "failed"
""",
        encoding="utf-8",
    )

    _, candidates = load_selection_protocol(path)

    assert candidates[0]["validation"]["macro_f1"] == 0.8
    assert candidates[1]["status"] == "failed"


def test_selection_protocol_can_load_candidates_before_summaries_exist(tmp_path) -> None:
    path = tmp_path / "protocol.toml"
    path.write_text(
        """
[protocol]
name = "fixture"

[[candidates]]
candidate_id = "head-1"
method = "head_only"
order = 0
config = "config.toml"
summary = "missing/summary.json"
""",
        encoding="utf-8",
    )

    _, candidates = load_selection_protocol(path, allow_missing_summaries=True)

    assert candidates[0]["summary_path"].endswith("missing\\summary.json")


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": 1, "kind": "other"}, "kind"),
        ({"schema_version": 1, "kind": "sentiment-selection"}, "test_access"),
    ],
)
def test_selection_manifest_rejects_invalid_payloads(tmp_path, payload, message) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_selection_manifest(path)


@pytest.mark.parametrize(
    "contents, message",
    [
        ("[protocol]\nname='x'", "protocol and candidates"),
        (
            "[protocol]\nname=''\n\n[[candidates]]\n"
            "candidate_id='x'\nmethod='head_only'\norder=0\n"
            "validation_macro_f1=0.5\nvalidation_loss=1.0",
            "non-empty",
        ),
        ("[protocol]\nname='x'\ntest_file='x'\n\n[[candidates]]\ncandidate_id='x'", "test"),
    ],
)
def test_selection_protocol_rejects_invalid_shapes(tmp_path, contents: str, message: str) -> None:
    path = tmp_path / "protocol.toml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_selection_protocol(path)
