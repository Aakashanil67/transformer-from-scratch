import hashlib
import json
from pathlib import Path

from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "playground.py"
V3_PROFILE = "Validation-selected v3 (exploratory)"
HISTORICAL_PROFILE = "Historical fixed profile"


def _record(path: Path, model: Path, *, digest: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "completed",
                "metrics": {"test": {"macro_f1": 0.8, "accuracy": 0.9}},
                "artifacts": {
                    "model": {"sha256": digest or hashlib.sha256(model.read_bytes()).hexdigest()}
                },
            }
        ),
        encoding="utf-8",
    )


def _comparison(path: Path, *, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "completed", "metrics": {"methods": {}}}
    if not malformed:
        payload["metrics"]["methods"]["lora"] = {
            "seeds": [17, 23, 41],
            "macro_f1": {"mean": 0.8, "standard_deviation": 0.01},
            "accuracy": {"mean": 0.9, "standard_deviation": 0.01},
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_root(tmp_path: Path, *, v3: str = "missing") -> Path:
    root = tmp_path / "app-root"
    historical_model = root / "artifacts/sentiment/tfidf/model.joblib"
    historical_model.parent.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(["profits rise", "losses widen", "shares steady"])
    classifier = LogisticRegression(max_iter=200).fit(features, ["positive", "negative", "neutral"])
    dump((vectorizer, classifier), historical_model)
    _record(
        root / "reports/results/financial-phrasebank-tfidf.json",
        historical_model,
    )
    _comparison(root / "reports/results/financial-phrasebank-comparison.json")

    if v3 == "valid":
        v3_model = root / "reports/results/v3/runs/baseline/model.joblib"
        v3_model.parent.mkdir(parents=True, exist_ok=True)
        dump((vectorizer, classifier), v3_model)
        _record(root / "reports/results/v3/runs/baseline.json", v3_model)
        v3_lora_model = root / "reports/results/v3/runs/lora-seed-17/model.pt"
        v3_lora_model.parent.mkdir(parents=True, exist_ok=True)
        dump((vectorizer, classifier), v3_lora_model)
        _record(root / "reports/results/v3/runs/lora-seed-17.json", v3_lora_model)
        _comparison(root / "reports/results/v3/comparison.json")
    elif v3 == "malformed":
        result = root / "reports/results/v3/runs/baseline.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("not json", encoding="utf-8")
    elif v3 == "hash-mismatch":
        v3_model = root / "reports/results/v3/runs/baseline/model.joblib"
        v3_model.parent.mkdir(parents=True, exist_ok=True)
        dump((vectorizer, classifier), v3_model)
        _record(root / "reports/results/v3/runs/baseline.json", v3_model, digest="0" * 64)
        _comparison(root / "reports/results/v3/comparison.json", malformed=True)
    return root


def _run(root: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv("TRANSFORMER_LAB_APP_ROOT", str(root))
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not at.exception
    return at


def test_app_defaults_to_v3_and_switches_profiles_by_key(tmp_path, monkeypatch):
    at = _run(_fixture_root(tmp_path), monkeypatch)

    assert at.selectbox(key="sentiment_profile").value == V3_PROFILE
    at.selectbox(key="sentiment_profile").set_value(HISTORICAL_PROFILE).run()

    assert at.selectbox(key="sentiment_profile").value == HISTORICAL_PROFILE
    assert at.selectbox(key="sentiment_method").value == "LoRA GPT-2"
    assert at.selectbox(key="sentiment_device").value == "auto"


def test_missing_v3_does_not_fall_back_to_historical_files(tmp_path, monkeypatch):
    at = _run(_fixture_root(tmp_path), monkeypatch)

    assert any("recorded result is missing" in item.value for item in at.info)
    assert not any("TF-IDF with balanced" in item.value for item in at.caption)


def test_historical_tfidf_profile_classifies_with_its_verified_model(tmp_path, monkeypatch):
    at = _run(_fixture_root(tmp_path), monkeypatch)
    at.selectbox(key="sentiment_profile").set_value(HISTORICAL_PROFILE)
    at.selectbox(key="sentiment_method").set_value("TF-IDF reference")
    at.text_area(key="sentiment_headline").set_value("profits rise")
    at.button(key="sentiment_classify").click().run(timeout=30)

    assert not at.exception
    assert any("Prediction:" in item.value for item in at.markdown)


def test_switching_profiles_uses_the_selected_verified_model(tmp_path, monkeypatch):
    root = _fixture_root(tmp_path, v3="valid")
    v3_model = root / "reports/results/v3/runs/baseline/model.joblib"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(["profits rise", "losses widen", "shares steady"])
    classifier = DummyClassifier(strategy="constant", constant="neutral").fit(
        features, ["positive", "negative", "neutral"]
    )
    dump((vectorizer, classifier), v3_model)
    _record(root / "reports/results/v3/runs/baseline.json", v3_model)

    at = _run(root, monkeypatch)
    at.selectbox(key="sentiment_method").set_value("TF-IDF reference")
    at.text_area(key="sentiment_headline").set_value("profits rise")
    at.button(key="sentiment_classify").click().run()
    assert any("Prediction: **neutral**" in item.value for item in at.markdown)

    at.selectbox(key="sentiment_profile").set_value(HISTORICAL_PROFILE)
    at.button(key="sentiment_classify").click().run()
    assert any("Prediction: **positive**" in item.value for item in at.markdown)


def test_malformed_v3_result_is_reported_without_crashing(tmp_path, monkeypatch):
    at = _run(_fixture_root(tmp_path, v3="malformed"), monkeypatch)
    at.selectbox(key="sentiment_method").set_value("TF-IDF reference").run()

    assert not at.exception
    assert any("could not be verified" in item.value for item in at.info)


def test_v3_hash_mismatch_is_not_hidden_by_historical_profile(tmp_path, monkeypatch):
    at = _run(_fixture_root(tmp_path, v3="hash-mismatch"), monkeypatch)
    at.selectbox(key="sentiment_method").set_value("TF-IDF reference").run()

    assert any("hash" in item.value for item in at.info)
    at.selectbox(key="sentiment_profile").set_value(HISTORICAL_PROFILE).run()
    assert not any("hash" in item.value for item in at.info)


def test_malformed_comparison_metrics_do_not_crash_or_create_caption(tmp_path, monkeypatch):
    root = _fixture_root(tmp_path, v3="valid")
    (root / "reports/results/v3/comparison.json").write_text(
        json.dumps({"status": "completed", "metrics": {"methods": {"lora": {}}}}),
        encoding="utf-8",
    )
    at = _run(root, monkeypatch)

    assert not at.exception
    assert not any("Three-seed macro-F1" in item.value for item in at.caption)


def test_malformed_nested_result_config_does_not_crash(tmp_path, monkeypatch):
    root = _fixture_root(tmp_path, v3="valid")
    result = root / "reports/results/v3/runs/lora-seed-17.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["config"] = "not an object"
    result.write_text(json.dumps(payload), encoding="utf-8")

    at = _run(root, monkeypatch)

    assert not at.exception
