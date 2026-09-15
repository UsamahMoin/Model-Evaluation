from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from framework.metrics import model_metrics
from framework.storage import read_results


def validate_complete(frame: pd.DataFrame):
    if frame.empty or not frame.run_complete.all():
        raise ValueError("Run is empty or incomplete")
    if frame.duplicated(["model", "test_id"]).any():
        raise ValueError("Duplicate model/test pairs")
    for model, group in frame.groupby("model"):
        if group.expected_count.nunique() != 1 or len(group) != group.expected_count.iloc[0]:
            raise ValueError(f"Missing rows or inconsistent expected count for {model}")


def compare(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    baseline_model: str | None = None,
    current_model: str | None = None,
) -> dict:
    validate_complete(baseline)
    validate_complete(current)
    if bool(baseline_model) != bool(current_model):
        raise ValueError("Specify both baseline_model and current_model for a model replacement")
    if baseline_model:
        pairs = [(baseline_model, current_model)]
    else:
        if set(baseline.model) != set(current.model):
            raise ValueError(
                "Model sets differ. Explicitly map --baseline-model and --current-model."
            )
        pairs = [(model, model) for model in sorted(baseline.model.unique())]
    reports = []
    for old_model, new_model in pairs:
        old = baseline[baseline.model == old_model].set_index("test_id")
        new = current[current.model == new_model].set_index("test_id")
        if old.empty or new.empty or set(old.index) != set(new.index):
            raise ValueError(f"Test coverage differs for {old_model} → {new_model}")
        new = new.loc[old.index]
        for column in ("scoring_hash", "category", "evaluator_version", "suite"):
            if not old[column].equals(new[column]):
                raise ValueError(
                    f"Incompatible {column}: compare runs with the same tests and scoring rules"
                )
        old_metrics = model_metrics(old.reset_index()).iloc[0]
        new_metrics = model_metrics(new.reset_index()).iloc[0]
        metrics = []
        for name in old_metrics.index:
            if name.endswith("_rate") or name in {"average_latency", "p95_latency"}:
                before, after = old_metrics[name], new_metrics[name]
                before = None if pd.isna(before) else float(before)
                after = None if pd.isna(after) else float(after)
                metrics.append(
                    {
                        "metric": name,
                        "baseline": before,
                        "current": after,
                        "delta": after - before
                        if before is not None and after is not None
                        else None,
                        "higher_is_better": name
                        not in {
                            "hallucination_rate",
                            "error_rate",
                            "average_latency",
                            "p95_latency",
                        },
                    }
                )
        reports.append(
            {
                "baseline_model": old_model,
                "current_model": new_model,
                "tests": len(old),
                "metrics": metrics,
                "new_failures": old.index[old.passed & ~new.passed].tolist(),
                "fixed_failures": old.index[~old.passed & new.passed].tolist(),
                "prompt_changes": int((old.prompt != new.prompt).sum()) if "prompt" in old else 0,
            }
        )
    return {
        "models": reports,
        "baseline_run": str(baseline.run_id.iloc[0]),
        "current_run": str(current.run_id.iloc[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare matched evaluation runs")
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--baseline-model")
    parser.add_argument("--current-model")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        report = compare(
            read_results(args.baseline),
            read_results(args.current),
            args.baseline_model,
            args.current_model,
        )
    except (ValueError, OSError) as exc:
        print(f"Comparison rejected: {exc}")
        return 2
    result = json.dumps(report, indent=2, allow_nan=False)
    print(result)
    if args.output:
        Path(args.output).write_text(result + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
