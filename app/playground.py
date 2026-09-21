"""Small local playground for the verified language and sentiment runs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import streamlit as st

from transformer_lab.commands import resolve_device
from transformer_lab.experiments.manifests import verify_run
from transformer_lab.inference.generation import load_local_generator
from transformer_lab.inference.sentiment import (
    classify_tfidf,
    load_local_sentiment_classifier,
    load_tfidf_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(os.environ.get("TRANSFORMER_LAB_APP_ROOT", str(ROOT))).expanduser().resolve()
GENERATION_ARTIFACT = APP_ROOT / "checkpoints" / "gpt2-small"
LORA_CHECKPOINTS = (
    APP_ROOT / "artifacts" / "sentiment" / "lora-seed-17" / "model.pt",
    APP_ROOT / "artifacts" / "sentiment" / "lora" / "model.pt",
)
TFIDF_CHECKPOINT = APP_ROOT / "artifacts" / "sentiment" / "tfidf" / "model.joblib"
TOKENIZER_DIRECTORY = GENERATION_ARTIFACT / "tokenizer"
PARITY_RESULT = APP_ROOT / "reports" / "results" / "gpt2-parity.json"
LORA_RESULT = APP_ROOT / "reports" / "results" / "financial-phrasebank-lora-seed-17.json"
TFIDF_RESULT = APP_ROOT / "reports" / "results" / "financial-phrasebank-tfidf.json"
COMPARISON_RESULT = APP_ROOT / "reports" / "results" / "financial-phrasebank-comparison.json"

PROFILES = {
    "Validation-selected v3 (exploratory)": {
        "lora_result": APP_ROOT / "reports/results/v3/runs/lora-seed-17.json",
        "lora_roots": (APP_ROOT / "reports/results/v3/runs/lora-seed-17",),
        "tfidf_result": APP_ROOT / "reports/results/v3/runs/baseline.json",
        "tfidf_roots": (APP_ROOT / "reports/results/v3/runs/baseline",),
        "comparison": APP_ROOT / "reports/results/v3/comparison.json",
    },
    "Historical fixed profile": {
        "lora_result": LORA_RESULT,
        "lora_roots": tuple(path.parent for path in LORA_CHECKPOINTS),
        "tfidf_result": TFIDF_RESULT,
        "tfidf_roots": (TFIDF_CHECKPOINT.parent,),
        "comparison": COMPARISON_RESULT,
    },
}


@st.cache_resource
def _generator(path: str, modified_ns: int, device_name: str):
    del modified_ns
    return load_local_generator(Path(path), resolve_device(device_name))


@st.cache_resource
def _sentiment_model(path: str, modified_ns: int):
    del modified_ns
    return load_tfidf_checkpoint(Path(path))


@st.cache_resource
def _transformer_sentiment_model(
    path: str, modified_ns: int, tokenizer_path: str, device_name: str
):
    del modified_ns
    return load_local_sentiment_classifier(
        Path(path), Path(tokenizer_path), resolve_device(device_name)
    )


def _result(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _completed_test_metrics(record: dict) -> dict | None:
    """Return test metrics only when a result record is complete and well-formed."""
    if record.get("status") != "completed":
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("test"), dict):
        return None
    test_metrics = metrics["test"]
    for key in ("macro_f1", "accuracy"):
        value = test_metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return None
    return test_metrics


def _comparison_summary(record: dict, method: str) -> dict | None:
    """Return a complete finite aggregate summary for one selected method."""
    if record.get("status") != "completed":
        return None
    metrics = record.get("metrics")
    methods = metrics.get("methods") if isinstance(metrics, dict) else None
    summary = methods.get(method) if isinstance(methods, dict) else None
    if not isinstance(summary, dict):
        return None
    seeds = summary.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        return None
    for metric_name in ("macro_f1", "accuracy"):
        metric = summary.get(metric_name)
        if not isinstance(metric, dict):
            return None
        mean = metric.get("mean")
        deviation = metric.get("standard_deviation")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, int | float)
            or not math.isfinite(float(mean))
            or not 0 <= float(mean) <= 1
        ):
            return None
        if len(seeds) == 1:
            if deviation is not None:
                return None
        elif (
            isinstance(deviation, bool)
            or not isinstance(deviation, int | float)
            or not math.isfinite(float(deviation))
            or float(deviation) < 0
        ):
            return None
    return summary


def _verified_artifact(
    result_path: Path, artifact_roots: tuple[Path, ...], artifact_name: str
) -> tuple[Path | None, str | None]:
    """Return only an artefact whose result-record hash verifies."""
    if not result_path.exists():
        return None, f"The recorded result is missing: {result_path}"
    errors: list[str] = []
    for artifact_root in artifact_roots:
        try:
            verified = verify_run(result_path, artifact_root=artifact_root)
        except (FileNotFoundError, OSError, ValueError) as error:
            errors.append(str(error))
            continue
        path = verified["artifact_paths"].get(artifact_name)
        if isinstance(path, Path):
            return path, None
    detail = errors[-1] if errors else "no matching artefact was declared"
    return None, f"The recorded {artifact_name} artefact could not be verified: {detail}"


st.set_page_config(page_title="Transformer lab", layout="centered")
st.title("Transformer lab")
st.caption("Inspect local GPT-2 generation and the measured Financial PhraseBank reference.")

tab_generation, tab_sentiment = st.tabs(["Generation", "Sentiment"])
with tab_generation:
    st.subheader("GPT-2 small")
    parity = _result(PARITY_RESULT)
    parity_metrics = parity.get("metrics") if parity.get("status") == "completed" else None
    revision = (
        parity.get("model", {}).get("revision") if isinstance(parity.get("model"), dict) else None
    )
    if (
        isinstance(parity_metrics, dict)
        and "max_absolute_error" in parity_metrics
        and isinstance(revision, str)
    ):
        st.caption(
            f"Pinned revision {revision[:12]}; "
            f"maximum parity error {parity_metrics['max_absolute_error']:.3g}."
        )
    elif isinstance(parity_metrics, dict) and "max_absolute_error" in parity_metrics:
        st.caption(f"Maximum parity error {parity_metrics['max_absolute_error']:.3g}.")
    elif parity:
        st.caption("GPT-2 parity has no completed result; run the parity check before generating.")
    prompt = st.text_area("Prompt", value="The market opened", key="generation_prompt")
    max_new_tokens = st.slider("New tokens", min_value=1, max_value=80, value=30)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    top_k = st.slider("Top-k", min_value=1, max_value=64, value=40)
    generation_device = st.selectbox(
        "Device", options=("auto", "cpu", "cuda"), key="generation_device"
    )
    seed = st.number_input("Seed", min_value=0, value=17, step=1)
    if st.button("Generate", type="primary"):
        model_path = GENERATION_ARTIFACT / "model.pt"
        if not model_path.exists():
            st.warning(
                "The local GPT-2 artefact is missing. Run the parity command once to prepare it."
            )
        elif not prompt:
            st.warning("Enter a prompt first.")
        else:
            try:
                generator = _generator(
                    str(GENERATION_ARTIFACT), model_path.stat().st_mtime_ns, generation_device
                )
                text = generator.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    seed=int(seed),
                    temperature=float(temperature),
                    top_k=int(top_k),
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                st.error(str(error))
            else:
                st.text_area("Generated text", value=text, height=220)

with tab_sentiment:
    st.subheader("Financial PhraseBank sentiment")
    profile_name = st.selectbox(
        "Experiment profile", options=tuple(PROFILES), key="sentiment_profile", index=0
    )
    profile = PROFILES[profile_name]
    method = st.selectbox(
        "Method", options=("LoRA GPT-2", "TF-IDF reference"), key="sentiment_method"
    )
    sentiment_device = st.selectbox(
        "Device", options=("auto", "cpu", "cuda"), key="sentiment_device"
    )
    transformer_method = method == "LoRA GPT-2"
    result_path = profile["lora_result"] if transformer_method else profile["tfidf_result"]
    artifact_roots = profile["lora_roots"] if transformer_method else profile["tfidf_roots"]
    checkpoint_path, checkpoint_error = _verified_artifact(result_path, artifact_roots, "model")
    sentiment_record = _result(result_path)
    comparison_record = _result(profile["comparison"])
    test_metrics = _completed_test_metrics(sentiment_record)
    if test_metrics is not None:
        model_label = (
            "GPT-2 small with rank-4 attention adapters"
            if transformer_method
            else "TF-IDF with balanced logistic regression"
        )
        config = sentiment_record.get("config")
        evaluation = config.get("evaluation") if isinstance(config, dict) else None
        seed = evaluation.get("seed") if isinstance(evaluation, dict) else None
        seed_label = f"seed {seed}" if isinstance(seed, int) else "recorded run"
        st.caption(
            f"{model_label}; {seed_label} test "
            f"macro-F1 {test_metrics['macro_f1']:.4f}, "
            f"accuracy {test_metrics['accuracy']:.4f}."
        )
        if transformer_method:
            summary = _comparison_summary(comparison_record, "lora")
            if summary is not None:
                mean = summary["macro_f1"]["mean"]
                deviation = summary["macro_f1"].get("standard_deviation")
                if deviation is not None:
                    st.caption(
                        f"{len(summary['seeds'])}-seed macro-F1 "
                        f"{float(mean):.4f} ± {float(deviation):.4f}."
                    )
        parameters = sentiment_record.get("parameters", {})
        if transformer_method and isinstance(parameters, dict):
            trainable = parameters.get("trainable")
            total = parameters.get("total")
        else:
            trainable = total = None
        if isinstance(trainable, int) and isinstance(total, int):
            st.caption(f"{trainable:,} of {total:,} parameters were trainable.")
    elif sentiment_record:
        status = sentiment_record.get("status", "invalid")
        st.caption(f"The recorded {method} run is {status}; rerun it before classifying.")
    if checkpoint_error:
        st.info(checkpoint_error)
    headline = st.text_area(
        "Financial headline",
        placeholder="Company raises its full-year guidance",
        key="sentiment_headline",
    )
    if st.button("Classify", type="primary", key="sentiment_classify"):
        if checkpoint_path is None:
            st.warning(
                f"The verified local {method} artefact is unavailable. "
                "Check the result and checkpoint files."
            )
        elif sentiment_record.get("status") != "completed":
            status = sentiment_record.get("status", "missing")
            st.warning(f"The recorded {method} run is {status}. Rerun it before classifying.")
        elif not headline.strip():
            st.warning("Enter a headline first.")
        else:
            try:
                if transformer_method:
                    classifier = _transformer_sentiment_model(
                        str(checkpoint_path),
                        checkpoint_path.stat().st_mtime_ns,
                        str(TOKENIZER_DIRECTORY),
                        sentiment_device,
                    )
                    result = classifier.classify(headline)
                else:
                    artifact = _sentiment_model(
                        str(checkpoint_path), checkpoint_path.stat().st_mtime_ns
                    )
                    result = classify_tfidf(artifact, headline)
            except (EOFError, FileNotFoundError, RuntimeError, ValueError) as error:
                st.error(str(error))
            else:
                st.write(f"Prediction: **{result['label']}**")
                st.write(f"Top-class probability: {result['top_class_probability']:.1%}")
                st.bar_chart(result["probabilities"])
