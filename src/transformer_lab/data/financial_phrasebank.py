"""Parse and split the licensed Financial PhraseBank archive without leakage."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

LABELS = frozenset({"negative", "neutral", "positive"})
_MEMBERS = {
    "50Agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    "66Agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "75Agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "allAgree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
}


@dataclass(frozen=True)
class Example:
    """One sentence and its human-assigned sentiment label."""

    text: str
    label: str
    group_id: str


@dataclass(frozen=True)
class PhraseBankSplits:
    """Deterministic, duplicate-safe partitions of the corpus."""

    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]


def normalise_for_grouping(text: str) -> str:
    """Canonicalise whitespace and Unicode only for duplicate grouping."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _group_id(text: str) -> str:
    return hashlib.sha256(normalise_for_grouping(text).encode("utf-8")).hexdigest()


def parse_phrasebank_lines(lines: Iterable[str]) -> tuple[Example, ...]:
    """Parse tab-separated sentence-label records from a PhraseBank member."""
    examples: list[Example] = []
    labels_by_group: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip("\r\n")
        if not line.strip():
            continue
        try:
            text, label = line.rsplit("@", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"line {line_number} has no sentence-label separator") from error
        text = text.strip()
        label = label.strip().casefold()
        if not text:
            raise ValueError(f"line {line_number} has an empty sentence")
        if label not in LABELS:
            raise ValueError(f"line {line_number} has unknown label '{label}'")
        group_id = _group_id(text)
        prior_label = labels_by_group.setdefault(group_id, label)
        if prior_label != label:
            raise ValueError(f"duplicate sentence has conflicting labels on line {line_number}")
        examples.append(Example(text=text, label=label, group_id=group_id))
    if not examples:
        raise ValueError("PhraseBank member contains no examples")
    return tuple(examples)


def load_phrasebank(archive: Path, subset: str = "75Agree") -> tuple[Example, ...]:
    """Load one agreement subset from a downloaded PhraseBank zip archive."""
    if subset not in _MEMBERS:
        choices = ", ".join(sorted(_MEMBERS))
        raise ValueError(f"subset must be one of: {choices}")
    with ZipFile(archive) as zipped:
        lines = zipped.read(_MEMBERS[subset]).decode("latin-1").splitlines()
    return parse_phrasebank_lines(lines)


def split_phrasebank(
    examples: Sequence[Example],
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PhraseBankSplits:
    """Split examples by duplicate group while approximately preserving labels."""
    if not examples:
        raise ValueError("examples must not be empty")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave test capacity")

    groups: dict[str, list[Example]] = defaultdict(list)
    labels: dict[str, str] = {}
    for example in examples:
        if example.label not in LABELS:
            raise ValueError(f"unknown label '{example.label}'")
        prior = labels.setdefault(example.group_id, example.label)
        if prior != example.label:
            raise ValueError(f"duplicate group {example.group_id} has conflicting labels")
        groups[example.group_id].append(example)

    by_label: dict[str, list[list[Example]]] = defaultdict(list)
    for group_id, members in groups.items():
        by_label[labels[group_id]].append(members)
    randomiser = random.Random(seed)
    if len(groups) < 3:
        raise ValueError("split requires at least three unique sentence groups")
    partition_groups: list[list[list[Example]]] = [[], [], []]
    fractions = (train_fraction, validation_fraction, 1 - train_fraction - validation_fraction)
    for label in sorted(by_label):
        label_groups = by_label[label]
        randomiser.shuffle(label_groups)
        label_total = sum(len(members) for members in label_groups)
        label_counts = [0, 0, 0]
        raw_targets = [label_total * fraction for fraction in fractions]
        label_targets = [int(target) for target in raw_targets]
        remainder = label_total - sum(label_targets)
        largest_remainders = sorted(range(3), key=lambda index: (-(raw_targets[index] % 1), index))
        for index in largest_remainders[:remainder]:
            label_targets[index] += 1
        for members in label_groups:
            deficits = [label_targets[index] - label_counts[index] for index in range(3)]
            target = max(range(3), key=lambda index: (deficits[index], -index))
            partition_groups[target].append(members)
            label_counts[target] += len(members)

    for empty in (index for index, members in enumerate(partition_groups) if not members):
        donor = max(range(3), key=lambda index: len(partition_groups[index]))
        candidate = min(
            range(len(partition_groups[donor])),
            key=lambda index: len(partition_groups[donor][index]),
        )
        partition_groups[empty].append(partition_groups[donor].pop(candidate))
    partitions = [
        [example for members in grouped_partition for example in members]
        for grouped_partition in partition_groups
    ]
    sorted_partitions = [
        tuple(sorted(partition, key=lambda item: (item.group_id, item.text)))
        for partition in partitions
    ]
    return PhraseBankSplits(
        train=sorted_partitions[0],
        validation=sorted_partitions[1],
        test=sorted_partitions[2],
    )


def split_manifest(splits: PhraseBankSplits) -> dict[str, list[dict[str, str]]]:
    """Return a redistributable manifest containing hashes and labels only."""
    return {
        name: [{"group_id": example.group_id, "label": example.label} for example in partition]
        for name, partition in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
    }


def evaluation_example_ids(examples: Sequence[Example]) -> list[str]:
    """Return stable row IDs using the duplicate group and occurrence order."""
    occurrences: Counter[str] = Counter()
    result: list[str] = []
    for example in examples:
        occurrence = occurrences[example.group_id]
        result.append(f"{example.group_id}:{occurrence}")
        occurrences[example.group_id] += 1
    return result


def split_summary(splits: PhraseBankSplits) -> dict[str, object]:
    """Return compact counts and fingerprints for a public result record."""
    manifest = split_manifest(splits)
    summary: dict[str, object] = {}
    for name, partition in manifest.items():
        encoded = json.dumps(partition, sort_keys=True, separators=(",", ":")).encode("utf-8")
        summary[name] = {
            "rows": len(partition),
            "class_counts": dict(Counter(item["label"] for item in partition)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    encoded_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary["sha256"] = hashlib.sha256(encoded_manifest).hexdigest()
    return summary
