from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSplit:
    train: str
    validation: str
    test: str


def split_document(
    text: str,
    *,
    validation_fraction: float,
    test_fraction: float,
    guard_chars: int,
) -> DocumentSplit:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("fractions must leave training capacity")
    if guard_chars < 0:
        raise ValueError("guard_chars must be non-negative")

    validation_size = int(len(text) * validation_fraction)
    test_size = int(len(text) * test_fraction)
    train_size = len(text) - validation_size - test_size - 2 * guard_chars
    if train_size <= 0:
        raise ValueError("split leaves no training text")

    validation_start = train_size + guard_chars
    test_start = validation_start + validation_size + guard_chars

    return DocumentSplit(
        train=text[:train_size],
        validation=text[validation_start : test_start - guard_chars],
        test=text[test_start:],
    )
