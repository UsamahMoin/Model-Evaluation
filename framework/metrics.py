import pandas as pd


def model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model"):
        ok = group[group.status == "ok"]
        hallucination = ok[ok.category == "hallucination"]
        row = {
            "model": model,
            "tests": len(group),
            "overall_pass_rate": float(group.passed.mean()),
            "average_latency": float(group.latency.mean()),
            "p95_latency": float(group.latency.quantile(0.95)),
            "error_rate": float((group.status != "ok").mean()),
            "hallucination_rate": float((~hallucination.passed).mean())
            if len(hallucination)
            else None,
            "hallucination_probes": len(hallucination),
        }
        for category, subset in group.groupby("category"):
            row[category + "_pass_rate"] = float(subset.passed.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def category_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["model", "category"], as_index=False).agg(
        pass_rate=("passed", "mean"),
        tests=("test_id", "size"),
        average_latency=("latency", "mean"),
    )


def failure_counts(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame.passed].groupby(["model", "failure_type"]).size().reset_index(name="count")
