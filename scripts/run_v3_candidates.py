"""Run the frozen v3 sentiment candidates using validation data only."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformer_lab.commands import resolve_device
from transformer_lab.config_io import ExperimentConfig, effective_config, load_experiment_config
from transformer_lab.experiments.manifests import config_digest, resolve_run
from transformer_lab.experiments.selection import (
    candidate_identity,
    load_selection_protocol,
    validate_candidate_summary,
)
from transformer_lab.experiments.sentiment import run_sentiment_candidate


def _candidate_config(
    config: ExperimentConfig,
    candidate: dict[str, Any],
    *,
    summary_path: Path,
    run_root: Path | None = None,
    selection_seed: int,
) -> tuple[ExperimentConfig, ExperimentConfig]:
    identity = candidate_identity(config, candidate, selection_seed=selection_seed)
    artifact_root = run_root or summary_path.parent
    run_config = replace(
        identity,
        output={
            **dict(identity.output),
            "directory": str(artifact_root / "run"),
            "result": str(artifact_root / "candidate-result.json"),
        },
    )
    return identity, run_config


def _write_failure(
    path: Path,
    *,
    candidate: dict[str, Any],
    identity: ExperimentConfig,
    error: BaseException,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "sentiment-candidate",
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "status": "failed",
        "effective_config": effective_config(identity),
        "config_digest": config_digest(identity),
        "data": {"selection_data": "validation_only"},
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_protocol(
    protocol_path: Path,
    *,
    device_name: str,
    resume: bool = False,
    refresh: bool = False,
    artifact_root: Path = Path("artifacts/selection/v3"),
) -> int:
    protocol, candidates = load_selection_protocol(
        protocol_path.resolve(), allow_missing_summaries=True
    )
    device = resolve_device(device_name)
    failures = 0
    selection_seed = protocol.get("selection_seed", 17)
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection protocol.selection_seed must be an integer")
    for index, candidate in enumerate(candidates, start=1):
        config_path = Path(str(candidate["config"]))
        if not config_path.is_absolute():
            config_path = (protocol_path.resolve().parent / config_path).resolve()
        summary_value = candidate.get("summary_path")
        if not isinstance(summary_value, str) or not summary_value:
            raise ValueError(f"candidate {candidate['candidate_id']} has no summary path")
        summary_path = Path(summary_value)
        if not summary_path.is_absolute():
            summary_path = (protocol_path.resolve().parent / summary_path).resolve()
        identity, run_config = _candidate_config(
            load_experiment_config(config_path),
            candidate,
            summary_path=summary_path,
            run_root=artifact_root.resolve() / str(candidate["candidate_id"]),
            selection_seed=selection_seed,
        )
        runtime_config = resolve_run(run_config, seed=selection_seed)
        runtime_artifact_root = Path(str(runtime_config.output["directory"]))
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as err:
                raise ValueError(f"invalid candidate summary: {summary_path}") from err
            validate_candidate_summary(
                payload,
                candidate=candidate,
                identity=identity,
                artifact_root=runtime_artifact_root,
            )
            if resume and not refresh:
                print(f"[{index}/{len(candidates)}] reusing {candidate['candidate_id']}")
                if payload.get("status") != "completed":
                    failures += 1
                continue
        print(f"[{index}/{len(candidates)}] running {candidate['candidate_id']}", flush=True)
        try:
            run_sentiment_candidate(
                run_config,
                device=device,
                summary_path=summary_path,
                seed=int(protocol.get("selection_seed", 17)),
                resume=resume,
                candidate_id=str(candidate["candidate_id"]),
                identity_config=identity,
            )
        except Exception as error:  # noqa: BLE001 - preserve every candidate outcome
            _write_failure(summary_path, candidate=candidate, identity=identity, error=error)
            failures += 1
            print(f"[{index}/{len(candidates)}] failed: {error}", flush=True)
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_candidate_summary(
            payload,
            candidate=candidate,
            identity=identity,
            artifact_root=runtime_artifact_root,
        )
        if payload.get("status") != "completed":
            failures += 1
        print(f"[{index}/{len(candidates)}] {payload.get('status', 'missing')}", flush=True)
    print(
        json.dumps(
            {"protocol": protocol["name"], "candidates": len(candidates), "failures": failures}
        )
    )
    return 2 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/selection/v3"))
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild terminal summaries from existing candidate checkpoints",
    )
    args = parser.parse_args(argv)
    return run_protocol(
        args.protocol,
        device_name=args.device,
        resume=args.resume,
        refresh=args.refresh,
        artifact_root=args.artifact_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
