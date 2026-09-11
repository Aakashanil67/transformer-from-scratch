from pathlib import Path

import pytest

from transformer_lab.data.download import download_file


def test_download_file_copies_a_file_url_to_a_nested_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "nested" / "download.txt"
    source.write_text("corpus", encoding="utf-8")

    downloaded = download_file(source.as_uri(), destination)

    assert downloaded == destination
    assert destination.read_text(encoding="utf-8") == "corpus"


def test_download_file_rejects_a_hash_mismatch_without_replacing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "download.txt"
    source.write_text("new corpus", encoding="utf-8")
    destination.write_text("known good corpus", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        download_file(source.as_uri(), destination, expected_sha256="0" * 64)

    assert destination.read_text(encoding="utf-8") == "known good corpus"
    assert not destination.with_suffix(".txt.part").exists()
