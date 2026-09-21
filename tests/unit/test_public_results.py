import json
import math
import statistics
from pathlib import Path


def _record_paths() -> dict[str, list[Path]]:
    root = Path(__file__).resolve().parents[2]
    runs = root / "reports/results/v3/runs"
    return {
        method: [runs / f"{method}-seed-{seed}.json" for seed in (17, 23, 41)]
        for method in ("full", "head_only", "lora")
    }


def test_v3_aggregate_replays_from_tracked_seed_records() -> None:
    root = Path(__file__).resolve().parents[2]
    comparison = json.loads(
        (root / "reports/results/v3/comparison.json").read_text(encoding="utf-8")
    )
    methods = comparison["metrics"]["methods"]
    for method, paths in _record_paths().items():
        values = [
            json.loads(path.read_text(encoding="utf-8"))["metrics"]["test"]["macro_f1"]
            for path in paths
        ]
        observed = methods[method]["macro_f1"]
        assert math.isclose(observed["mean"], statistics.mean(values), rel_tol=0, abs_tol=1e-15)
        assert math.isclose(
            observed["standard_deviation"], statistics.stdev(values), rel_tol=0, abs_tol=1e-15
        )

    assert all(
        comparison_item["bootstrap_ci"]["lower"] <= 0 <= comparison_item["bootstrap_ci"]["upper"]
        for comparison_item in comparison["metrics"]["paired_comparisons"]["full_minus_lora"]
    )
