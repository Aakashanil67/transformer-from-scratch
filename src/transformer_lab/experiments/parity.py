"""Compare the local decoder against a pinned public GPT-2 checkpoint."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lab.config_io import ExperimentConfig, effective_config
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import environment_metadata
from transformer_lab.gpt2 import REVISION, gpt2_small_config, load_huggingface_gpt2_weights
from transformer_lab.models.transformer import DecoderOnlyTransformer

MODEL_ID = "openai-community/gpt2"
PROMPTS = (
    "Hello",
    "The market opened",
    "Revenue rose 12.4% in 2024.",
    "Café prices changed.",
    "The board met on Monday. The discussion continued after lunch.",
)


def run(config: ExperimentConfig, *, device: torch.device | None = None) -> Path:
    """Run the fixed prompt suite and write a tracked parity result."""
    device = device or torch.device("cpu")
    model_id = str(config.model.get("base_model", MODEL_ID))
    revision = str(config.model.get("model_revision", REVISION))
    tolerance = float(config.model.get("tolerance", 1e-4))
    prompts = tuple(config.model.get("prompts", PROMPTS))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    reference = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to(device).eval()
    model = DecoderOnlyTransformer(gpt2_small_config()).to(device).eval()
    load_report = load_huggingface_gpt2_weights(model, reference.state_dict())
    prompt_results = []
    started = time.perf_counter()
    with torch.no_grad():
        for prompt in prompts:
            token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            reference_logits = reference(token_ids).logits.float()
            local_logits, _ = model(token_ids)
            difference = (reference_logits - local_logits.float()).abs()
            reference_next = reference_logits[:, -1, :].argmax(dim=-1)
            local_next = local_logits[:, -1, :].argmax(dim=-1)
            prompt_results.append(
                {
                    "prompt": prompt,
                    "tokens": int(token_ids.size(1)),
                    "max_absolute_error": float(difference.max().item()),
                    "mean_absolute_error": float(difference.mean().item()),
                    "next_token_agreement": bool(torch.equal(reference_next, local_next)),
                }
            )
    max_error = max(item["max_absolute_error"] for item in prompt_results)
    mean_error = sum(item["mean_absolute_error"] for item in prompt_results) / len(prompt_results)
    result_path = Path(config.output.get("result", "reports/results/gpt2-parity.json"))
    passed = max_error <= tolerance
    measured = {
        "tolerance": tolerance,
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "next_token_agreement": sum(item["next_token_agreement"] for item in prompt_results)
        / len(prompt_results),
        "prompts": prompt_results,
    }
    record = ExperimentRecord(
        run_id=config.experiment["name"],
        status="completed" if passed else "failed",
        model={
            "id": model_id,
            "revision": revision,
            "tokenizer_id": model_id,
            "tokenizer_revision": revision,
            "loader_tensors": load_report.loaded,
        },
        config=effective_config(config),
        data={"prompt_count": len(prompts), "seed": config.evaluation.get("seed", 17)},
        environment=environment_metadata(device),
        timing={"seconds": time.perf_counter() - started},
        metrics=measured if passed else None,
        error=(
            None
            if passed
            else {
                "type": "ParityError",
                "message": f"max absolute error {max_error} exceeded tolerance {tolerance}",
            }
        ),
    )
    record.write(result_path)
    if not passed:
        raise RuntimeError(f"GPT-2 parity failed: max error {max_error} > tolerance {tolerance}")
    artifact_dir = Path(config.output.get("directory", "checkpoints/gpt2-small"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary = artifact_dir / "model.pt.tmp"
    torch.save(
        {
            "schema_version": 1,
            "architecture": asdict(model.config),
            "state_dict": model.state_dict(),
            "model_id": model_id,
            "revision": revision,
        },
        temporary,
    )
    temporary.replace(artifact_dir / "model.pt")
    tokenizer.save_pretrained(artifact_dir / "tokenizer")
    return result_path
