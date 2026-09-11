"""Local generation helpers with no network access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from transformer_lab.config import GPTConfig
from transformer_lab.models.bigram import BigramLanguageModel
from transformer_lab.models.transformer import DecoderOnlyTransformer


@dataclass
class LocalGenerator:
    model: DecoderOnlyTransformer
    tokenizer: Any
    device: torch.device

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float,
        top_k: int | None,
        do_sample: bool = True,
    ) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")
        token_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        if token_ids.size(1) > self.model.config.block_size:
            raise ValueError("prompt exceeds the model context window")
        output = self.model.generate(
            token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        return self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)


def load_local_generator(artifact_dir: Path, device: torch.device) -> LocalGenerator:
    model_path = artifact_dir / "model.pt"
    tokenizer_dir = artifact_dir / "tokenizer"
    if not model_path.exists():
        raise FileNotFoundError(f"local generation artifact is missing {model_path.name}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError("local generation tokenizer is missing")
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported local generation artifact schema")
    model = DecoderOnlyTransformer(GPTConfig(**payload["architecture"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    return LocalGenerator(model=model, tokenizer=tokenizer, device=device)


def generate_from_bigram(
    checkpoint: Path,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
    device: torch.device | None = None,
) -> str:
    """Generate UTF-8 text from a locally saved byte-level bigram checkpoint."""
    device = device or torch.device("cpu")
    if not checkpoint.exists():
        raise FileNotFoundError(f"generation checkpoint does not exist: {checkpoint}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("model_type") != "bigram":
        raise ValueError("generation checkpoint is not a bigram model")
    model = BigramLanguageModel(int(payload["vocab_size"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    return bytes(generated[0].tolist()).decode("utf-8", errors="replace")
