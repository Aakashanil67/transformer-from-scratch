"""Compare the local decoder against a pinned public GPT-2 checkpoint."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lab.config_io import ExperimentConfig, effective_config
from transformer_lab.experiments.records import ExperimentRecord
from transformer_lab.experiments.runtime import environment_metadata, file_sha256, source_provenance
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


def random_token_cases(lengths: list[int], *, vocab_size: int, seed: int) -> list[torch.Tensor]:
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("random parity lengths must be positive")
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (1, length), generator=generator) for length in lengths]


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
    random_results = []
    batch_results = []
    hidden_errors: list[float] = []
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
        lengths = [int(length) for length in config.model.get("random_token_lengths", [])]
        seed = int(config.evaluation.get("seed", 17))
        for token_ids in (
            random_token_cases(lengths, vocab_size=model.config.vocab_size, seed=seed)
            if lengths
            else []
        ):
            token_ids = token_ids.to(device)
            reference_output = reference(token_ids, output_hidden_states=True)
            local_logits, _ = model(token_ids)
            difference = (reference_output.logits.float() - local_logits.float()).abs()
            layer_errors = [
                float((expected.float() - observed.float()).abs().max().item())
                for expected, observed in zip(
                    reference_output.hidden_states, model.layer_states(token_ids), strict=True
                )
            ]
            hidden_errors.extend(layer_errors)
            random_results.append(
                {
                    "tokens": int(token_ids.size(1)),
                    "max_absolute_error": float(difference.max().item()),
                    "mean_absolute_error": float(difference.mean().item()),
                    "max_hidden_state_error": max(layer_errors),
                    "next_token_agreement": bool(
                        torch.equal(
                            reference_output.logits[:, -1].argmax(dim=-1),
                            local_logits[:, -1].argmax(dim=-1),
                        )
                    ),
                }
            )
        batch_prompts = list(config.model.get("batch_prompts", []))
        if batch_prompts:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            token_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            reference_logits = reference(token_ids, attention_mask=attention_mask).logits.float()
            local_logits, _ = model(token_ids)
            valid = attention_mask.bool().unsqueeze(-1).expand_as(local_logits)
            difference = (reference_logits - local_logits.float()).abs()[valid]
            agreements = []
            for row, length in enumerate(attention_mask.sum(dim=1).tolist()):
                index = int(length) - 1
                agreements.append(
                    bool(
                        torch.equal(
                            reference_logits[row, index].argmax(),
                            local_logits[row, index].argmax(),
                        )
                    )
                )
            batch_results.append(
                {
                    "batch_size": len(batch_prompts),
                    "token_lengths": [int(length) for length in attention_mask.sum(dim=1)],
                    "max_absolute_error": float(difference.max().item()),
                    "mean_absolute_error": float(difference.mean().item()),
                    "next_token_agreement": sum(agreements) / len(agreements),
                }
            )
    cases = [*prompt_results, *random_results, *batch_results]
    max_error = max(item["max_absolute_error"] for item in cases)
    mean_error = sum(item["mean_absolute_error"] for item in cases) / len(cases)
    result_path = Path(config.output.get("result", "reports/results/gpt2-parity.json"))
    hidden_tolerance = float(config.model.get("hidden_state_tolerance", tolerance))
    max_hidden_error = max(hidden_errors, default=0.0)
    passed = max_error <= tolerance and max_hidden_error <= hidden_tolerance
    measured = {
        "tolerance": tolerance,
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "next_token_agreement": sum(float(item["next_token_agreement"]) for item in cases)
        / len(cases),
        "prompts": prompt_results,
        "random_token_cases": random_results,
        "batched_cases": batch_results,
        "hidden_state_tolerance": hidden_tolerance,
        "max_hidden_state_error": max_hidden_error,
    }
    artifact_dir = Path(config.output.get("directory", "checkpoints/gpt2-small"))
    artifacts = {}
    if passed:
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
        model_path = artifact_dir / "model.pt"
        temporary.replace(model_path)
        tokenizer.save_pretrained(artifact_dir / "tokenizer")
        artifacts = {"model": {"sha256": file_sha256(model_path)}}
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
        data={
            "prompt_count": len(prompts),
            "random_case_count": len(random_results),
            "batch_case_count": len(batch_results),
            "seed": config.evaluation.get("seed", 17),
        },
        environment=environment_metadata(device),
        provenance=source_provenance(config.source_path),
        artifacts=artifacts,
        timing={"seconds": time.perf_counter() - started},
        metrics=measured if passed else None,
        error=(
            None
            if passed
            else {
                "type": "ParityError",
                "message": (
                    f"parity error exceeded a tolerance: logits {max_error}/{tolerance}; "
                    f"hidden states {max_hidden_error}/{hidden_tolerance}"
                ),
            }
        ),
    )
    record.write(result_path)
    if not passed:
        raise RuntimeError(record.error["message"])
    return result_path
