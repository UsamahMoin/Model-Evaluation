"""Prepare blind human review sheets and measure agreement after real labeling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from framework.storage import read_results, write_csv

KEYS = ["run_id", "model", "test_id"]


def review_sample(results: pd.DataFrame, count: int = 40) -> pd.DataFrame:
    if count < 1:
        raise ValueError("Sample size must be positive")
    eligible = results[results.status == "ok"]
    if eligible.empty:
        raise ValueError("No successful generations available to label")
    # Round-robin across model/category strata avoids a single category dominating.
    groups = [
        group.sample(frac=1, random_state=42)
        for _, group in eligible.groupby(["model", "category"])
    ]
    sample = []
    for index in range(max(map(len, groups))):
        for group in groups:
            if index < len(group) and len(sample) < count:
                sample.append(group.iloc[index])
    frame = pd.DataFrame(sample)[
        KEYS + ["category", "prompt", "expected_answer", "response"]
    ].reset_index(drop=True)
    frame["human_passed"] = ""
    frame["human_reason"] = ""
    return frame


def agreement(results: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required = set(KEYS + ["human_passed"])
    if not required <= set(labels):
        raise ValueError(f"Labels need columns {sorted(required)}")
    if labels.duplicated(KEYS).any():
        raise ValueError("Duplicate human labels")
    if labels.empty:
        raise ValueError("No human labels supplied")
    human = labels.human_passed.astype(str).str.strip().str.lower()
    if not human.isin(["true", "false", "pass", "fail", ""]).all():
        raise ValueError("human_passed must be PASS/FAIL or true/false; blanks remain unlabeled")
    labels = labels.copy()
    labels["human_passed"] = human.map({"true": True, "false": False, "pass": True, "fail": False})
    matched = labels.merge(
        results, on=KEYS, how="left", validate="one_to_one", indicator=True, suffixes=("_label", "")
    )
    if (matched._merge != "both").any():
        raise ValueError("Human labels contain keys absent from this run")
    labeled = matched[matched.human_passed.notna()].copy()
    if labeled.empty:
        raise ValueError("No completed human labels yet")
    if "judge_passed" not in labeled:
        raise ValueError("Run has no judge verdicts")
    judge_values = labeled.judge_passed.astype(str).str.lower()
    judged = labeled[judge_values.isin(["true", "false"])].copy()
    excluded = len(labeled) - len(judged)
    if judged.empty:
        raise ValueError("No valid judge verdicts for the labeled responses; run with --judge")
    judged["judge_passed"] = judged.judge_passed.astype(str).str.lower() == "true"
    humans = judged.human_passed.astype(bool)
    judges = judged.judge_passed.astype(bool)
    observed = float(accuracy_score(humans, judges))
    # Wilson interval for raw agreement (descriptive; correlated cases violate iid assumptions).
    n, z = len(judged), 1.96
    center = (observed + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    kappa = float(cohen_kappa_score(humans, judges)) if len(set(humans) | set(judges)) > 1 else None
    report = {
        "labeled": len(labeled),
        "compared": n,
        "excluded_missing_judge": excluded,
        "agreement": observed,
        "agreement_95pct_wilson": [center - margin, center + margin],
        "cohen_kappa": kappa if kappa is not None and math.isfinite(kappa) else None,
        "confusion_matrix_labels": ["FAIL", "PASS"],
        "confusion_matrix_human_rows_judge_columns": confusion_matrix(
            humans, judges, labels=[False, True]
        ).tolist(),
        "caution": "Small, stratified review sample; agreement does not establish judge correctness. Wilson interval assumes independent samples.",
    }
    disagreements = judged[humans != judges].drop(columns="_merge")
    return report, disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("results")
    sample.add_argument("--count", type=int, default=40)
    sample.add_argument("--output", default="results/human_labels.csv")
    score = commands.add_parser("score")
    score.add_argument("results")
    score.add_argument("labels")
    score.add_argument("--output-dir", default="results/calibration")
    args = parser.parse_args()
    try:
        frame = read_results(args.results)
        if args.command == "sample":
            if Path(args.output).exists():
                raise ValueError(
                    "Review file already exists; choose a new output to preserve human labels"
                )
            write_csv(review_sample(frame, args.count), args.output)
            print(f"Blind review sheet saved: {args.output}. Fill human_passed with PASS/FAIL.")
        else:
            labels = pd.read_csv(args.labels, keep_default_na=False)
            report, disagreements = agreement(frame, labels)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "agreement.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            write_csv(disagreements, out / "disagreements.csv")
            print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Calibration stopped: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
