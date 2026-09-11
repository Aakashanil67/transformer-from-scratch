import hashlib
import json
from pathlib import Path

from transformer_lab.data.provenance import write_provenance


def test_write_provenance_records_source_and_file_hashes(tmp_path: Path) -> None:
    raw_file = tmp_path / "raw.txt"
    prepared_file = tmp_path / "prepared.txt"
    manifest_file = tmp_path / "provenance.json"
    raw_file.write_text("source text", encoding="utf-8")
    prepared_file.write_text("prepared text", encoding="utf-8")

    write_provenance(
        manifest_file,
        source={"url": "https://example.test/source", "revision": "2026-08-25"},
        files={"raw": raw_file, "prepared": prepared_file},
    )

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "revision": "2026-08-25",
        "url": "https://example.test/source",
    }
    assert manifest["files"]["raw"] == {
        "bytes": len(b"source text"),
        "sha256": hashlib.sha256(b"source text").hexdigest(),
    }
    assert manifest["files"]["prepared"] == {
        "bytes": len(b"prepared text"),
        "sha256": hashlib.sha256(b"prepared text").hexdigest(),
    }
