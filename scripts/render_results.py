"""Render and check deterministic tables derived from experiment JSON."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

from transformer_lab.evaluation.reporting import render_comparison_markdown

ROOT = Path(__file__).resolve().parents[1]


def _load_comparison(results_dir: Path) -> dict[str, Any]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read result JSON: {path}") from error
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if isinstance(metrics, dict) and isinstance(metrics.get("methods"), dict):
            payloads.append((path, payload))
    if len(payloads) != 1:
        raise ValueError(
            "expected exactly one aggregate comparison JSON in "
            f"{results_dir}, found {len(payloads)}"
        )
    payload = copy.deepcopy(payloads[0][1])
    method_slugs = {"tfidf": "tfidf", "head_only": "head-only", "lora": "lora", "full": "full"}
    for method, summary in payload["metrics"]["methods"].items():
        slug = method_slugs.get(str(method))
        if slug is None:
            continue
        records = (
            [results_dir / "financial-phrasebank-tfidf.json"]
            if slug == "tfidf"
            else [
                results_dir / f"financial-phrasebank-{slug}-seed-{seed}.json"
                for seed in summary["seeds"]
            ]
        )
        memory_values: list[int] = []
        for record_path in records:
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            memory = record.get("memory", {}).get("peak_cuda_allocated_bytes")
            if isinstance(memory, int):
                memory_values.append(memory)
        if memory_values:
            summary["peak_allocated_bytes"] = max(memory_values)
    return payload


def render(results_dir: Path) -> str:
    """Return the checked aggregate table for a results directory."""
    return render_comparison_markdown(_load_comparison(results_dir), title="Results")


def _svg_document(title: str, body: str, *, width: int = 760, height: int = 460) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{html.escape(title)}</title>",
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="30" y="30" font-family="sans-serif" font-size="20" '
            f'fill="#16212b">{html.escape(title)}</text>',
            body,
            "</svg>",
        ]
    )


def _scatter_figure(results_dir: Path) -> str:
    colours = {"full": "#b23a48", "lora": "#247ba0", "head_only": "#f18f01", "baseline": "#2a9d8f"}
    points: list[tuple[str, int, float]] = []
    for path in sorted(results_dir.glob("financial-phrasebank-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {}).get("test", {})
        parameters = payload.get("parameters", {})
        mode = payload.get("model", {}).get("mode", "baseline")
        if isinstance(metrics, dict) and isinstance(parameters, dict):
            if isinstance(metrics.get("macro_f1"), (int, float)) and isinstance(
                parameters.get("trainable"), int
            ):
                points.append((str(mode), parameters["trainable"], float(metrics["macro_f1"])))
    if not points:
        raise ValueError("no sentiment run records available for parameter figure")
    left, top, width, height = 80, 60, 620, 330
    x_values = [math.log10(max(point[1], 1)) for point in points]
    x_min, x_max = min(x_values), max(x_values)

    def x(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * width

    def y(value: float) -> float:
        return top + (1.0 - value) * height

    body = [
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" '
        f'y2="{top + height}" stroke="#16212b"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#16212b"/>',
        f'<text x="{left + width / 2:.0f}" y="435" text-anchor="middle" '
        'font-family="sans-serif">log10(trainable parameters)</text>',
        '<text x="18" y="235" transform="rotate(-90 18 235)" '
        'text-anchor="middle" font-family="sans-serif">test macro-F1</text>',
    ]
    for mode, parameters, macro_f1 in points:
        colour = colours.get(mode, "#555")
        cx, cy = x(math.log10(max(parameters, 1))), y(macro_f1)
        body.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colour}"/>'
            f'<text x="{cx + 7:.1f}" y="{cy - 7:.1f}" font-family="sans-serif" '
            f'font-size="11">{html.escape(mode)}</text>'
        )
    return _svg_document("Per-seed macro-F1 versus trainable parameters", "\n".join(body))


def _scratch_figure(results_dir: Path) -> str:
    path = results_dir / "tiny-shakespeare-transformer-scratch.json"
    if not path.exists():
        raise ValueError(f"scratch transformer result is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("metrics", {}).get("training_loss")
    validation = payload.get("metrics", {}).get("validation_loss")
    steps = payload.get("metrics", {}).get("validation_steps")
    if (
        not isinstance(training, list)
        or not isinstance(validation, list)
        or not isinstance(steps, list)
    ):
        raise ValueError("scratch result has no recorded learning curves")
    maximum = max(float(value) for value in training + validation)

    def point(step: float, loss: float) -> str:
        return f"{80 + step / max(len(training), 1) * 620:.1f},{390 - loss / maximum * 330:.1f}"

    training_points = " ".join(
        point(index * 10 + 1, float(value)) for index, value in enumerate(training[::10])
    )
    validation_points = " ".join(
        point(float(step), float(value)) for step, value in zip(steps, validation, strict=True)
    )
    body = [
        '<line x1="80" y1="390" x2="700" y2="390" stroke="#16212b"/>',
        '<line x1="80" y1="60" x2="80" y2="390" stroke="#16212b"/>',
        '<text x="390" y="435" text-anchor="middle" font-family="sans-serif">microstep</text>',
        '<text x="18" y="235" transform="rotate(-90 18 235)" '
        'text-anchor="middle" font-family="sans-serif">cross-entropy loss</text>',
        f'<polyline points="{training_points}" fill="none" stroke="#247ba0" stroke-width="2"/>',
        f'<polyline points="{validation_points}" fill="none" stroke="#b23a48" stroke-width="2"/>',
        '<text x="560" y="80" font-family="sans-serif" fill="#247ba0">training</text>',
        '<text x="560" y="100" font-family="sans-serif" fill="#b23a48">validation</text>',
    ]
    return _svg_document("Scratch transformer learning curve", "\n".join(body))


def _calibration_figure(results_dir: Path) -> str:
    """Render pre/post-temperature reliability from a verified local checkpoint."""
    record_path = results_dir / "financial-phrasebank-lora-seed-17.json"
    artifact_root = ROOT / "artifacts" / "sentiment" / "lora-seed-17"
    if not record_path.exists():
        raise ValueError("LoRA seed-17 result is required for calibration figure")
    from transformer_lab.experiments.manifests import verify_run

    verified = verify_run(record_path, artifact_root=artifact_root)
    model_path = verified["artifact_paths"].get("model")
    evaluation_path = verified["artifact_paths"].get("evaluation")
    if not isinstance(model_path, Path) or not isinstance(evaluation_path, Path):
        raise ValueError("verified LoRA result has no model/evaluation artefacts")
    import torch

    from transformer_lab.config import GPTConfig
    from transformer_lab.evaluation.classification import apply_temperature
    from transformer_lab.lora import LoRAConfig, inject_lora
    from transformer_lab.models.sentiment import SentimentClassifier

    record = json.loads(record_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = SentimentClassifier(
        GPTConfig(**checkpoint["architecture"]), num_labels=int(checkpoint["num_labels"])
    )
    if checkpoint["mode"] == "lora":
        inject_lora(model.backbone, LoRAConfig(**checkpoint["lora"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    frozen = torch.load(evaluation_path, map_location="cpu", weights_only=True)
    targets = frozen.get("targets")
    if not isinstance(targets, torch.Tensor):
        raise ValueError("evaluation artefact has no targets")
    with torch.no_grad():
        logits, _ = model(frozen["input_ids"], attention_mask=frozen["attention_mask"])
    raw = logits.softmax(dim=-1).numpy()
    temperature = float(record["metrics"]["calibration"]["validation_temperature"])
    calibrated = apply_temperature(logits.numpy(), temperature)
    target_values = targets.numpy()

    def bins(probabilities: Any) -> list[tuple[float, float, int]]:
        confidence, predictions = probabilities.max(axis=1), probabilities.argmax(axis=1)
        output: list[tuple[float, float, int]] = []
        for index in range(10):
            lower, upper = index / 10, (index + 1) / 10
            mask = (confidence >= lower) & ((confidence < upper) | (index == 9))
            if mask.any():
                output.append(
                    (
                        float(confidence[mask].mean()),
                        float((predictions[mask] == target_values[mask]).mean()),
                        int(mask.sum()),
                    )
                )
        return output

    before, after = bins(raw), bins(calibrated)
    body = [
        '<line x1="80" y1="390" x2="700" y2="390" stroke="#16212b"/>',
        '<line x1="80" y1="60" x2="80" y2="390" stroke="#16212b"/>',
        '<line x1="80" y1="390" x2="700" y2="60" stroke="#9aa6b2" stroke-dasharray="5,5"/>',
        '<text x="390" y="435" text-anchor="middle" '
        'font-family="sans-serif">mean confidence</text>',
        '<text x="18" y="235" transform="rotate(-90 18 235)" '
        'text-anchor="middle" font-family="sans-serif">accuracy</text>',
        '<text x="520" y="80" font-family="sans-serif" fill="#b23a48">before</text>',
        '<text x="520" y="100" font-family="sans-serif" fill="#247ba0">after</text>',
    ]
    for series, colour in ((before, "#b23a48"), (after, "#247ba0")):
        for confidence_value, accuracy, count in series:
            cx, cy = 80 + confidence_value * 620, 390 - accuracy * 330
            body.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colour}"/>'
                f'<text x="{cx + 7:.1f}" y="{cy - 7:.1f}" font-family="sans-serif" '
                f'font-size="10">n={count}</text>'
            )
    return _svg_document(
        "LoRA seed-17 calibration reliability before and after scaling", "\n".join(body)
    )


def render_figures(results_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_contents = {
        "macro-f1-vs-parameters.svg": _scatter_figure(results_dir),
        "calibration-reliability.svg": _calibration_figure(results_dir),
        "scratch-transformer-learning-curve.svg": _scratch_figure(results_dir),
    }
    paths = []
    for name, content in figure_contents.items():
        path = figures_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.results / "rendered-results.md"
    try:
        rendered = render(args.results.resolve())
        if args.check:
            if not output.exists():
                raise ValueError(f"generated report is missing: {output}")
            observed = output.read_text(encoding="utf-8")
            if observed != rendered:
                raise ValueError(f"generated report is stale: {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(f"Wrote {output}")
            if args.figures:
                for figure in render_figures(args.results.resolve(), args.figures.resolve()):
                    print(f"Wrote {figure}")
    except (OSError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
