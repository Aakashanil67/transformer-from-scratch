import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Tokenizer

from transformer_lab.config import GPTConfig
from transformer_lab.inference.generation import generate_from_bigram, load_local_generator
from transformer_lab.models.bigram import BigramLanguageModel
from transformer_lab.models.transformer import DecoderOnlyTransformer


def test_generation_service_uses_a_local_checkpoint(tmp_path: Path) -> None:
    model = BigramLanguageModel(256)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model_type": "bigram", "vocab_size": 256, "state_dict": model.state_dict()},
        checkpoint,
    )

    output = generate_from_bigram(
        checkpoint, "Hi", max_new_tokens=3, seed=2, temperature=0.8, top_k=8
    )

    assert output.startswith("Hi")


def test_generation_service_sampling_controls_are_seeded(tmp_path: Path) -> None:
    model = BigramLanguageModel(256)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model_type": "bigram", "vocab_size": 256, "state_dict": model.state_dict()},
        checkpoint,
    )

    first = generate_from_bigram(
        checkpoint, "A", max_new_tokens=4, seed=7, temperature=0.5, top_k=3
    )
    second = generate_from_bigram(
        checkpoint, "A", max_new_tokens=4, seed=7, temperature=0.5, top_k=3
    )

    assert first == second


def test_local_transformer_generator_loads_model_and_tokenizer_without_network(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "gpt2"
    tokenizer_dir = artifact / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "vocab.json").write_text(
        json.dumps({"a": 0, "b": 1, "<|endoftext|>": 2}), encoding="utf-8"
    )
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = GPT2Tokenizer(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
        unk_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(tokenizer_dir)
    config = GPTConfig(vocab_size=3, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = DecoderOnlyTransformer(config)
    torch.save(
        {"schema_version": 1, "architecture": config.__dict__, "state_dict": model.state_dict()},
        artifact / "model.pt",
    )

    generator = load_local_generator(artifact, torch.device("cpu"))
    output = generator.generate(
        "ab", max_new_tokens=2, seed=7, temperature=1.0, top_k=2, do_sample=False
    )

    assert output.startswith("ab")


def test_local_transformer_generator_rejects_missing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "missing"

    try:
        load_local_generator(artifact, torch.device("cpu"))
    except FileNotFoundError as error:
        assert "model.pt" in str(error)
    else:
        raise AssertionError("missing local model was accepted")


def test_local_transformer_generator_rejects_incomplete_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "incomplete"
    (artifact / "tokenizer").mkdir(parents=True)
    torch.save({"schema_version": 1}, artifact / "model.pt")

    with pytest.raises(ValueError, match="incomplete"):
        load_local_generator(artifact, torch.device("cpu"))
