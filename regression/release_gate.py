import argparse
import json
from pathlib import Path

from framework.metrics import model_metrics
from framework.schema import CATEGORIES, GateSettings, load_config
from framework.storage import read_results
from regression.compare_runs import compare, validate_complete


def check_gates(
    current, settings: GateSettings, baseline=None, baseline_model=None, current_model=None
):
    reasons = []
    try:
        validate_complete(current)
        report = (
            compare(baseline, current, baseline_model, current_model)
            if baseline is not None
            else None
        )
    except ValueError as exc:
        return {"passed": False, "reasons": [str(exc)]}
    evaluated = current[current.model == current_model] if current_model else current
    if evaluated.empty:
        return {"passed": False, "reasons": ["No rows for the selected model"]}
    for model, group in evaluated.groupby("model"):
        if "dataset_complete" in group and not group.dataset_complete.all():
            reasons.append(f"{model}: smoke-test subset cannot establish release readiness")
        missing = set(CATEGORIES) - set(group.category)
        if missing:
            reasons.append(f"{model}: missing required categories {sorted(missing)}")
        if (group.status != "ok").any():
            reasons.append(f"{model}: generation or evaluation errors present")
        for column in ("judge_error", "semantic_error"):
            if column in group and group[column].fillna("").ne("").any():
                reasons.append(f"{model}: {column} present")
        metrics = model_metrics(group).iloc[0]
        limits = [
            ("overall_pass_rate", settings.overall_pass_rate_min, True),
            ("instruction_following_pass_rate", settings.instruction_following_min, True),
            ("structured_output_pass_rate", settings.structured_output_min, True),
            ("hallucination_rate", settings.hallucination_rate_max, False),
        ]
        for name, threshold, minimum in limits:
            value = metrics.get(name)
            if value is None or value != value:
                reasons.append(f"{model}: {name} is unavailable")
            elif (minimum and value + 1e-12 < threshold) or (
                not minimum and value - 1e-12 > threshold
            ):
                reasons.append(
                    f"{model}: {name} {value:.1%} must be {'at least' if minimum else 'at most'} {threshold:.1%}"
                )
    if report:
        for model in report["models"]:
            for metric in model["metrics"]:
                if metric["metric"] in {category + "_pass_rate" for category in CATEGORIES}:
                    if (
                        metric["delta"] is not None
                        and metric["delta"] < -settings.max_category_drop - 1e-12
                    ):
                        reasons.append(
                            f"{model['current_model']}: {metric['metric']} dropped {-metric['delta'] * 100:.1f} percentage points (limit {settings.max_category_drop * 100:.1f})"
                        )
    return {"passed": not reasons, "reasons": reasons, "regression_checked": baseline is not None}


def main():
    parser = argparse.ArgumentParser(
        description="Fail closed on quality, coverage, and regression violations"
    )
    parser.add_argument("current")
    parser.add_argument("--baseline")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml")
    )
    parser.add_argument("--baseline-model")
    parser.add_argument("--current-model")
    args = parser.parse_args()
    try:
        result = check_gates(
            read_results(args.current),
            load_config(args.config).release_gates,
            read_results(args.baseline) if args.baseline else None,
            args.baseline_model,
            args.current_model,
        )
    except (ValueError, OSError) as exc:
        result = {"passed": False, "reasons": [str(exc)]}
    print("RELEASE GATE: " + ("PASS" if result["passed"] else "FAILED"))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
