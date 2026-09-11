import json
from pathlib import Path

import torch
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import GPT2Tokenizer

from transformer_lab.config import GPTConfig
from transformer_lab.inference.sentiment import (
    classify_tfidf,
    classify_with_tfidf,
    load_local_sentiment_classifier,
    load_tfidf_checkpoint,
)
from transformer_lab.models.sentiment import SentimentClassifier


def test_sentiment_service_returns_label_and_probabilities(tmp_path: Path) -> None:
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(["profit rises", "loss widens", "board meets"])
    classifier = LogisticRegression(max_iter=1000).fit(
        features, ["positive", "negative", "neutral"]
    )
    checkpoint = tmp_path / "model.joblib"
    dump((vectorizer, classifier), checkpoint)

    result = classify_with_tfidf(checkpoint, "profit rises")

    assert result["label"] in {"negative", "neutral", "positive"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
    assert result["confidence"] == max(result["probabilities"].values())


def test_loaded_tfidf_checkpoint_can_be_reused_for_multiple_headlines(tmp_path: Path) -> None:
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(["profit rises", "loss widens", "board meets"])
    classifier = LogisticRegression(max_iter=1000).fit(
        features, ["positive", "negative", "neutral"]
    )
    checkpoint = tmp_path / "model.joblib"
    dump((vectorizer, classifier), checkpoint)
    artifact = load_tfidf_checkpoint(checkpoint)

    first = classify_tfidf(artifact, "profit rises")
    second = classify_tfidf(artifact, "loss widens")

    assert first["label"] == "positive"
    assert second["label"] == "negative"


def test_local_transformer_sentiment_classifier_is_offline(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "vocab.json").write_text(
        json.dumps({"a": 0, "b": 1, "<|endoftext|>": 2}), encoding="utf-8"
    )
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = GPT2Tokenizer(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
        unk_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(tokenizer_dir)
    config = GPTConfig(vocab_size=3, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = SentimentClassifier(config, num_labels=3)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": config.__dict__,
            "state_dict": model.state_dict(),
            "mode": "head_only",
            "labels": ["negative", "neutral", "positive"],
            "num_labels": 3,
        },
        checkpoint,
    )

    classifier = load_local_sentiment_classifier(
        checkpoint, tokenizer_dir, torch.device("cpu"), max_length=4
    )
    result = classifier.classify("ab")

    assert result["label"] in {"negative", "neutral", "positive"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
