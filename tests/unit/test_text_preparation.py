import json
from pathlib import Path

from transformer_lab.data.text import prepare_text_corpus


def test_prepare_text_corpus_writes_splits_and_provenance(tmp_path: Path) -> None:
    raw_file = tmp_path / "source.txt"
    output_dir = tmp_path / "prepared"
    raw_file.write_text("abcdefghijklmnopqrst", encoding="utf-8")

    prepared = prepare_text_corpus(
        raw_file,
        output_dir,
        source={"url": "https://example.test/source", "revision": "v1"},
        validation_fraction=0.25,
        test_fraction=0.25,
        guard_chars=2,
    )

    assert prepared.train.read_text(encoding="utf-8") == "abcdef"
    assert prepared.validation.read_text(encoding="utf-8") == "ijklm"
    assert prepared.test.read_text(encoding="utf-8") == "pqrst"

    provenance = json.loads(prepared.provenance.read_text(encoding="utf-8"))
    assert provenance["source"]["url"] == "https://example.test/source"
    assert set(provenance["files"]) == {"raw", "test", "train", "validation"}
