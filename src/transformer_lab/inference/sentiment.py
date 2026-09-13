"""Local sentiment inference helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from joblib import load
from transformers import AutoTokenizer

from transformer_lab.config import GPTConfig
from transformer_lab.lora import LoRAConfig, inject_lora
from transformer_lab.models.sentiment import SentimentClassifier


@dataclass
class LocalSentimentClassifier:
    model: SentimentClassifier
    tokenizer: Any
    labels: tuple[str, ...]
    device: torch.device
    max_length: int
    calibration_temperature: float = 1.0

    def classify(self, text: str) -> dict[str, object]:
        if not text.strip():
            raise ValueError("text must not be empty")
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits, _ = self.model(
                encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            )
            probabilities = (
                (logits / self.calibration_temperature).softmax(dim=-1)[0].cpu().tolist()
            )
        scores = dict(zip(self.labels, map(float, probabilities), strict=True))
        label = max(scores, key=scores.__getitem__)
        return {
            "label": label,
            "top_class_probability": scores[label],
            "probabilities": scores,
        }


def load_local_sentiment_classifier(
    checkpoint: Path,
    tokenizer_dir: Path,
    device: torch.device,
    *,
    max_length: int = 64,
) -> LocalSentimentClassifier:
    if not checkpoint.exists():
        raise FileNotFoundError(f"sentiment checkpoint does not exist: {checkpoint}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError("local sentiment tokenizer is missing")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("sentiment artifact is incomplete: expected a mapping")
    schema_version = payload.get("schema_version", 1)
    if schema_version not in {1, 2}:
        raise ValueError("sentiment artifact has an unsupported schema")
    missing = {"architecture", "num_labels", "state_dict", "labels", "mode"} - payload.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"sentiment artifact is incomplete: missing {names}")
    labels = payload["labels"]
    num_labels = payload["num_labels"]
    if (
        not isinstance(labels, (list, tuple))
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("sentiment artifact labels must be a unique non-empty list")
    if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels != len(labels):
        raise ValueError("sentiment artifact label count does not match labels")
    if payload["mode"] not in {"head_only", "lora", "full"}:
        raise ValueError("sentiment artifact has an unsupported mode")
    temperature = payload.get("calibration_temperature", 1.0)
    if (
        not isinstance(temperature, int | float)
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("sentiment artifact temperature must be finite and positive")
    stored_max_length = payload.get("max_length", max_length)
    if (
        isinstance(stored_max_length, bool)
        or not isinstance(stored_max_length, int)
        or stored_max_length <= 0
    ):
        raise ValueError("sentiment artifact max_length must be a positive integer")
    for key in ("tokenizer_id", "tokenizer_revision"):
        if key in payload and (not isinstance(payload[key], str) or not payload[key]):
            raise ValueError(f"sentiment artifact {key} must be a non-empty string")
    try:
        model = SentimentClassifier(GPTConfig(**payload["architecture"]), num_labels=num_labels)
        if payload.get("mode") == "lora":
            if "lora" not in payload:
                raise ValueError("missing lora configuration")
            inject_lora(model.backbone, LoRAConfig(**payload["lora"]))
        model.load_state_dict(payload["state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("sentiment artifact is incompatible with the model schema") from error
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LocalSentimentClassifier(
        model=model,
        tokenizer=tokenizer,
        labels=tuple(labels),
        device=device,
        max_length=stored_max_length,
        calibration_temperature=float(temperature),
    )


def load_tfidf_checkpoint(checkpoint: Path) -> tuple[Any, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"sentiment checkpoint does not exist: {checkpoint}")
    artifact = load(checkpoint)
    if not isinstance(artifact, tuple) or len(artifact) != 2:
        raise ValueError("sentiment checkpoint must contain a vectorizer and classifier")
    return artifact


def classify_tfidf(artifact: tuple[Any, Any], text: str) -> dict[str, object]:
    if not text.strip():
        raise ValueError("text must not be empty")
    vectorizer, classifier = artifact
    features = vectorizer.transform([text])
    label = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    scores = {
        str(name): float(probability)
        for name, probability in zip(classifier.classes_, probabilities, strict=True)
    }
    return {
        "label": str(label),
        "top_class_probability": max(scores.values()),
        "probabilities": scores,
    }


def classify_with_tfidf(checkpoint: Path, text: str) -> dict[str, object]:
    """Classify one headline with the locally saved reference model."""
    return classify_tfidf(load_tfidf_checkpoint(checkpoint), text)
