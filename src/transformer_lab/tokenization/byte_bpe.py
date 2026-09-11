"""A small byte-level BPE tokenizer for inspection and experimentation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Self


def most_frequent_pair(token_ids: list[int]) -> tuple[int, int] | None:
    """Return the most common adjacent pair, resolving ties by token IDs."""
    pairs = Counter(zip(token_ids, token_ids[1:], strict=False))
    if not pairs:
        return None
    return min(pairs, key=lambda pair: (-pairs[pair], pair))


def _merge_pair(token_ids: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(token_ids):
        if index < len(token_ids) - 1 and tuple(token_ids[index : index + 2]) == pair:
            merged.append(replacement)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


@dataclass(frozen=True)
class ByteBPETokenizer:
    """A byte-level BPE tokenizer with merges learned from one UTF-8 corpus."""

    merges: dict[tuple[int, int], int]
    vocabulary: dict[int, bytes]

    @classmethod
    def train(cls, text: str, *, vocab_size: int) -> Self:
        """Learn up to ``vocab_size - 256`` merge rules from ``text``."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
        token_ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}

        while len(vocabulary) < vocab_size:
            pair = most_frequent_pair(token_ids)
            if pair is None:
                break
            new_token_id = len(vocabulary)
            merges[pair] = new_token_id
            vocabulary[new_token_id] = vocabulary[pair[0]] + vocabulary[pair[1]]
            token_ids = _merge_pair(token_ids, pair, new_token_id)

        return cls(merges=merges, vocabulary=vocabulary)

    def encode(self, text: str) -> list[int]:
        """Convert UTF-8 text to token IDs using learned merges in training order."""
        token_ids = list(text.encode("utf-8"))
        for pair, replacement in self.merges.items():
            token_ids = _merge_pair(token_ids, pair, replacement)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to UTF-8 text."""
        return b"".join(self.vocabulary[token_id] for token_id in token_ids).decode("utf-8")

    def save(self, destination: Path) -> None:
        """Save merge rules in a compact JSON format."""
        payload = {"version": 1, "merges": [list(pair) for pair in self.merges]}
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, source: Path) -> Self:
        """Load a tokenizer saved by :meth:`save`."""
        payload = json.loads(source.read_text(encoding="utf-8"))
        vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
        merges: dict[tuple[int, int], int] = {}
        for index, raw_pair in enumerate(payload["merges"]):
            pair = (int(raw_pair[0]), int(raw_pair[1]))
            replacement = 256 + index
            merges[pair] = replacement
            vocabulary[replacement] = vocabulary[pair[0]] + vocabulary[pair[1]]
        return cls(merges=merges, vocabulary=vocabulary)
