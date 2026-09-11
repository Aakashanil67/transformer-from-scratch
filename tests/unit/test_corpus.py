import pytest

from transformer_lab.data.corpus import split_document


def test_split_document_discards_guard_regions_between_sets() -> None:
    split = split_document(
        "abcdefghijklmnopqrst",
        validation_fraction=0.25,
        test_fraction=0.25,
        guard_chars=2,
    )

    assert split.train == "abcdef"
    assert split.validation == "ijklm"
    assert split.test == "pqrst"


def test_split_document_rejects_a_split_without_training_text() -> None:
    with pytest.raises(ValueError, match="split leaves no training text"):
        split_document(
            "abcdefgh",
            validation_fraction=0.4,
            test_fraction=0.4,
            guard_chars=2,
        )


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction", "guard_chars", "message"),
    [
        (-0.1, 0.1, 0, "fractions must be non-negative"),
        (0.1, -0.1, 0, "fractions must be non-negative"),
        (0.6, 0.4, 0, "fractions must leave training capacity"),
        (0.1, 0.1, -1, "guard_chars must be non-negative"),
    ],
)
def test_split_document_rejects_invalid_split_parameters(
    validation_fraction: float,
    test_fraction: float,
    guard_chars: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        split_document(
            "abcdefghijklmnopqrst",
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            guard_chars=guard_chars,
        )
