"""Fetch and prepare Tiny Shakespeare for language-model experiments."""

from pathlib import Path

from transformer_lab.data.download import download_file
from transformer_lab.data.text import prepare_text_corpus

SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
)
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    raw_file = repository_root / "data" / "raw" / "tiny_shakespeare.txt"
    prepared_dir = repository_root / "data" / "prepared" / "tiny_shakespeare"
    download_file(SOURCE_URL, raw_file, expected_sha256=SOURCE_SHA256)
    prepared = prepare_text_corpus(
        raw_file,
        prepared_dir,
        source={"name": "Tiny Shakespeare", "url": SOURCE_URL},
        validation_fraction=0.05,
        test_fraction=0.05,
        guard_chars=1_024,
    )
    print(f"Prepared corpus at {prepared.train.parent}")


if __name__ == "__main__":
    main()
