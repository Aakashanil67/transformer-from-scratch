from collections import Counter

import pytest

from transformer_lab.data.financial_phrasebank import (
    evaluation_example_ids,
    parse_phrasebank_lines,
    split_manifest,
    split_phrasebank,
    split_summary,
)


def test_parser_preserves_text_and_groups_exact_duplicates() -> None:
    examples = parse_phrasebank_lines(
        ["Café sales rise@positive", "Café sales rise@positive", "Costs hold@neutral"]
    )

    assert examples[0].text == "Café sales rise"
    assert examples[0].group_id == examples[1].group_id


def test_parser_rejects_conflicting_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="conflicting labels"):
        parse_phrasebank_lines(["Same sentence@positive", "Same sentence@negative"])


def test_split_keeps_duplicate_groups_in_one_partition() -> None:
    examples = parse_phrasebank_lines(
        [
            "A@positive",
            "A@positive",
            "B@positive",
            "C@negative",
            "D@negative",
            "E@neutral",
            "F@neutral",
        ]
    )

    splits = split_phrasebank(examples, seed=17)
    membership = {}
    partitions = (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    )
    for name, partition in partitions:
        for example in partition:
            membership.setdefault(example.group_id, set()).add(name)

    assert all(len(partitions) == 1 for partitions in membership.values())
    assert set(split_manifest(splits)) == {"train", "validation", "test"}


def test_grouped_split_preserves_each_label_when_groups_allow_it() -> None:
    examples = parse_phrasebank_lines(
        [
            f"{label} sentence {index}@{label}"
            for label in sorted(("negative", "neutral", "positive"))
            for index in range(30)
        ]
    )

    splits = split_phrasebank(examples, seed=3)

    for partition in (splits.train, splits.validation, splits.test):
        counts = Counter(example.label for example in partition)
        assert len(set(counts.values())) == 1
    assert len(splits.train) == 63
    assert sorted((len(splits.validation), len(splits.test))) == [12, 15]


def test_split_summary_is_compact_and_content_addressed() -> None:
    splits = split_phrasebank(
        parse_phrasebank_lines(
            [
                "Loss widened@negative",
                "Profit rose@positive",
                "Sales held@neutral",
                "Costs fell@positive",
                "Outlook held@neutral",
                "Debt rose@negative",
            ]
        ),
        seed=17,
    )

    summary = split_summary(splits)

    assert set(summary) == {"train", "validation", "test", "sha256"}
    assert len(summary["sha256"]) == 64
    assert all("sha256" in summary[name] for name in ("train", "validation", "test"))
    assert "Profit rose" not in repr(summary)


def test_evaluation_example_ids_keep_duplicate_occurrences_distinct() -> None:
    examples = parse_phrasebank_lines(["Repeated@positive", "Repeated@positive", "Other@neutral"])

    ids = evaluation_example_ids(examples)

    assert ids[0].endswith(":0")
    assert ids[1].endswith(":1")
    assert ids[0].split(":")[0] == ids[1].split(":")[0]
    assert ids[2].endswith(":0")
