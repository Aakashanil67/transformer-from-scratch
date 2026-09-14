"""Reproducible TF-IDF reference model for Financial PhraseBank."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformer_lab.config_io import ExperimentConfig, effective_config
from transformer_lab.data.financial_phrasebank import (
    evaluation_example_ids,
    load_phrasebank,
    split_phrasebank,
    split_summary,
)
from transformer_lab.evaluation.classification import classification_metrics, multiclass_log_loss
from transformer_lab.experiments.manifests import build_manifest, config_digest, resolve_run
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import environment_metadata, file_sha256, source_provenance

REPOSITORY = "financial_phrasebank"
REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
ARCHIVE = "data/FinancialPhraseBank-v1.0.zip"
ARCHIVE_MEMBER = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
SEED = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(revision: str = REVISION) -> tuple[list[str], list[str]]:
    """Download the pinned archive and return sentence/label lists."""
    archive = hf_hub_download(REPOSITORY, ARCHIVE, repo_type="dataset", revision=revision)
    examples = load_phrasebank(Path(archive), "75Agree")
    return [example.text for example in examples], [example.label for example in examples]


def run_baseline(
    config: ExperimentConfig | None = None,
    *,
    validation_only: bool = False,
    summary_path: Path | None = None,
    identity_config: ExperimentConfig | None = None,
    candidate_id: str | None = None,
) -> Path:
    """Train the TF-IDF baseline, optionally stopping after validation."""
    if config is None:
        raise ValueError("a baseline configuration is required")
    config = resolve_run(config)
    revision = str(config.data.get("dataset_revision", REVISION))
    archive_path = Path(
        hf_hub_download(REPOSITORY, ARCHIVE, repo_type="dataset", revision=revision)
    )
    examples = load_phrasebank(archive_path, str(config.data.get("subset", "75Agree")))
    seed = int(config.evaluation.get("seed", SEED))
    split_seed = int(config.data.get("split_seed", SEED))
    splits = split_phrasebank(examples, seed=split_seed)
    train_text = [example.text for example in splits.train]
    validation_text = [example.text for example in splits.validation]
    train_labels = [example.label for example in splits.train]
    validation_labels = [example.label for example in splits.validation]
    if not validation_only:
        test_text = [example.text for example in splits.test]
        test_labels = [example.label for example in splits.test]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    classifier = LogisticRegression(
        C=float(config.optimization.get("c", 1.0)),
        max_iter=2_000,
        class_weight="balanced",
        random_state=seed,
    )
    started = perf_counter()
    classifier.fit(vectorizer.fit_transform(train_text), train_labels)
    elapsed_seconds = perf_counter() - started
    validation_predictions = classifier.predict(vectorizer.transform(validation_text))
    validation_probabilities = classifier.predict_proba(vectorizer.transform(validation_text))
    labels = ["negative", "neutral", "positive"]
    validation_metrics = classification_metrics(
        np.asarray(validation_labels),
        np.asarray(validation_predictions),
        labels=labels,
        probabilities=validation_probabilities,
        bootstrap_samples=int(config.evaluation.get("bootstrap_samples", 1_000)),
        seed=seed,
    )
    validation_metrics["loss"] = multiclass_log_loss(
        np.asarray(validation_labels),
        validation_probabilities,
        labels=labels,
    )
    model_path = Path(config.output.get("directory", "artifacts/sentiment/tfidf")) / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump((vectorizer, classifier), model_path)
    if validation_only:
        destination = summary_path or model_path.with_name("validation-summary.json")
        identity = identity_config or config
        payload = {
            "schema_version": 1,
            "kind": "sentiment-candidate",
            "candidate_id": candidate_id or str(config.experiment["name"]),
            "method": "baseline",
            "status": "completed",
            "effective_config": effective_config(identity),
            "config_digest": config_digest(identity),
            "validation": validation_metrics,
            "validation_candidates": [{"epoch": 1, "metrics": validation_metrics}],
            "data": {
                "repository": REPOSITORY,
                "revision": revision,
                "subset": config.data.get("subset", "75Agree"),
                "train_rows": len(splits.train),
                "validation_rows": len(splits.validation),
            },
            "timing": {
                "train_seconds": elapsed_seconds,
                "examples_per_second": len(splits.train) / elapsed_seconds,
            },
            "artifacts": {"model": {"path": model_path.name, "sha256": file_sha256(model_path)}},
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination
    test_features = vectorizer.transform(test_text)
    test_predictions = classifier.predict(test_features)
    test_probabilities = classifier.predict_proba(test_features)
    test_metrics = classification_metrics(
        np.asarray(test_labels),
        np.asarray(test_predictions),
        labels=labels,
        probabilities=test_probabilities,
        bootstrap_samples=int(config.evaluation.get("bootstrap_samples", 1_000)),
        seed=seed,
    )
    result_path = Path(config.output["result"])
    evaluation_path = model_path.with_name("evaluation.joblib")
    dump(
        {
            "features": test_features,
            "targets": np.asarray(test_labels),
            "predictions": np.asarray(test_predictions),
            "target_ids": np.asarray([labels.index(label) for label in test_labels]),
            "prediction_ids": np.asarray([labels.index(label) for label in test_predictions]),
            "probabilities": np.asarray(test_probabilities),
            "labels": labels,
            "example_ids": np.asarray(evaluation_example_ids(splits.test)),
        },
        evaluation_path,
    )
    provenance = source_provenance(config.source_path)
    artifacts = {
        "model": {"path": "model.joblib", "sha256": file_sha256(model_path)},
        "evaluation": {"path": "evaluation.joblib", "sha256": file_sha256(evaluation_path)},
    }
    record = ExperimentRecord(
        run_id=config.experiment["name"],
        status="completed",
        model={"type": "tfidf-logistic", "vectorizer": "word-1-2gram"},
        config=effective_config(config),
        data={
            "repository": REPOSITORY,
            "revision": revision,
            "archive_member": ARCHIVE_MEMBER,
            "subset": config.data.get("subset", "75Agree"),
            "archive_sha256": _sha256(archive_path),
            "examples": len(examples),
            "unique_sentence_groups": len({example.group_id for example in examples}),
            "class_counts": dict(Counter(example.label for example in examples)),
            "split": split_summary(splits),
        },
        environment=environment_metadata(torch.device("cpu")),
        provenance=provenance,
        artifacts=artifacts,
        manifest=build_manifest(
            config,
            provenance=provenance,
            data_identity={"split": split_summary(splits)},
            evaluation={
                "seed": seed,
                "bootstrap_samples": config.evaluation.get("bootstrap_samples", 1_000),
            },
            artifacts=artifacts,
        ),
        parameters={
            "total": int(classifier.coef_.size + classifier.intercept_.size),
            "trainable": int(classifier.coef_.size + classifier.intercept_.size),
            "trainable_share": 1.0,
        },
        timing={
            "train_seconds": elapsed_seconds,
            "examples_per_second": len(splits.train) / elapsed_seconds,
        },
        metrics={"validation": validation_metrics, "test": test_metrics},
        schema_version=3,
    )
    record.write(result_path)
    return result_path
