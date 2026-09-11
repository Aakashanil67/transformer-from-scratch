"""Small local playground for the verified language and sentiment runs."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from transformer_lab.commands import resolve_device
from transformer_lab.inference.generation import load_local_generator
from transformer_lab.inference.sentiment import (
    classify_tfidf,
    load_local_sentiment_classifier,
    load_tfidf_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATION_ARTIFACT = ROOT / "checkpoints" / "gpt2-small"
LORA_CHECKPOINT = ROOT / "artifacts" / "sentiment" / "lora" / "model.pt"
TFIDF_CHECKPOINT = ROOT / "artifacts" / "sentiment" / "tfidf" / "model.joblib"
TOKENIZER_DIRECTORY = GENERATION_ARTIFACT / "tokenizer"
PARITY_RESULT = ROOT / "reports" / "results" / "gpt2-parity.json"
LORA_RESULT = ROOT / "reports" / "results" / "financial-phrasebank-lora-seed-17.json"
TFIDF_RESULT = ROOT / "reports" / "results" / "financial-phrasebank-tfidf.json"


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
    return metrics["test"]


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
    device_name = st.selectbox("Device", options=("auto", "cpu", "cuda"))
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
                    str(GENERATION_ARTIFACT), model_path.stat().st_mtime_ns, device_name
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
    method = st.selectbox("Method", options=("LoRA GPT-2", "TF-IDF reference"))
    transformer_method = method == "LoRA GPT-2"
    result_path = LORA_RESULT if transformer_method else TFIDF_RESULT
    checkpoint_path = LORA_CHECKPOINT if transformer_method else TFIDF_CHECKPOINT
    sentiment_record = _result(result_path)
    test_metrics = _completed_test_metrics(sentiment_record)
    if test_metrics is not None:
        model_label = (
            "GPT-2 small with rank-4 attention adapters"
            if transformer_method
            else "TF-IDF with balanced logistic regression"
        )
        st.caption(
            f"{model_label}; "
            f"test macro-F1 {test_metrics['macro_f1']:.4f}, "
            f"accuracy {test_metrics['accuracy']:.4f}."
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
    headline = st.text_area(
        "Financial headline",
        placeholder="Company raises its full-year guidance",
        key="sentiment_headline",
    )
    if st.button("Classify", type="primary"):
        if not checkpoint_path.exists():
            st.warning(
                f"The local {method} artefact is missing. Run its sentiment experiment first."
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
                        device_name,
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
                st.write(f"Confidence: {result['confidence']:.1%}")
                st.bar_chart(result["probabilities"])
