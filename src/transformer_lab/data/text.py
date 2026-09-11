"""Prepare a contiguous text corpus for autoregressive language modelling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from transformer_lab.data.corpus import split_document
from transformer_lab.data.provenance import write_provenance


@dataclass(frozen=True)
class PreparedTextCorpus:
    train: Path
    validation: Path
    test: Path
    provenance: Path


def prepare_text_corpus(
    raw_file: Path,
    output_dir: Path,
    *,
    source: Mapping[str, str],
    validation_fraction: float,
    test_fraction: float,
    guard_chars: int,
) -> PreparedTextCorpus:
    """Split a UTF-8 document and write the files used by a language-model run."""
    split = split_document(
        raw_file.read_text(encoding="utf-8"),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        guard_chars=guard_chars,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.txt"
    validation_file = output_dir / "validation.txt"
    test_file = output_dir / "test.txt"
    provenance_file = output_dir / "provenance.json"

    train_file.write_text(split.train, encoding="utf-8")
    validation_file.write_text(split.validation, encoding="utf-8")
    test_file.write_text(split.test, encoding="utf-8")
    write_provenance(
        provenance_file,
        source=source,
        files={
            "raw": raw_file,
            "train": train_file,
            "validation": validation_file,
            "test": test_file,
        },
    )
    return PreparedTextCorpus(
        train=train_file,
        validation=validation_file,
        test=test_file,
        provenance=provenance_file,
    )
