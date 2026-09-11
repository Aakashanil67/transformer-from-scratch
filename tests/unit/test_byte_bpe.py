from pathlib import Path

import pytest

from transformer_lab.tokenization.byte_bpe import ByteBPETokenizer, most_frequent_pair


def test_most_frequent_pair_breaks_frequency_ties_lexicographically() -> None:
    token_ids = [10, 11, 10, 11, 12, 13, 12, 13]

    assert most_frequent_pair(token_ids) == (10, 11)


def test_byte_bpe_round_trips_unicode_text() -> None:
    text = "Café U0001f642 Café"
    tokenizer = ByteBPETokenizer.train(text, vocab_size=260)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_bpe_uses_learned_merges_when_encoding() -> None:
    tokenizer = ByteBPETokenizer.train("aaaaaa", vocab_size=257)

    assert len(tokenizer.merges) == 1
    assert len(tokenizer.encode("aaaaaa")) == 3


def test_byte_bpe_save_and_load_preserve_merge_order(tmp_path: Path) -> None:
    tokenizer = ByteBPETokenizer.train("banana bandana", vocab_size=262)
    tokenizer_file = tmp_path / "tokenizer.json"

    tokenizer.save(tokenizer_file)
    restored = ByteBPETokenizer.load(tokenizer_file)

    assert restored.merges == tokenizer.merges
    assert restored.encode("banana") == tokenizer.encode("banana")
    assert restored.decode(restored.encode("banana")) == "banana"


@pytest.mark.parametrize("vocab_size", [0, 255])
def test_byte_bpe_requires_space_for_all_byte_tokens(vocab_size: int) -> None:
    with pytest.raises(ValueError, match="vocab_size must be at least 256"):
        ByteBPETokenizer.train("text", vocab_size=vocab_size)
