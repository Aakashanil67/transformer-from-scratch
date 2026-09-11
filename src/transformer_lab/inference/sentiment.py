"""Local sentiment inference helpers."""

from __future__ import annotations

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
            probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
        scores = dict(zip(self.labels, map(float, probabilities), strict=True))
        label = max(scores, key=scores.__getitem__)
        return {"label": label, "confidence": scores[label], "probabilities": scores}


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
    model = SentimentClassifier(
        GPTConfig(**payload["architecture"]), num_labels=int(payload["num_labels"])
    )
    if payload.get("mode") == "lora":
        inject_lora(model.backbone, LoRAConfig(**payload["lora"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LocalSentimentClassifier(
        model=model,
        tokenizer=tokenizer,
        labels=tuple(payload["labels"]),
        device=device,
        max_length=max_length,
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
        "confidence": max(scores.values()),
        "probabilities": scores,
    }


def classify_with_tfidf(checkpoint: Path, text: str) -> dict[str, object]:
    """Classify one headline with the locally saved reference model."""
    return classify_tfidf(load_tfidf_checkpoint(checkpoint), text)
