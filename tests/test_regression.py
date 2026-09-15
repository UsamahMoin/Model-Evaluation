import pandas as pd
import pytest

from framework.metrics import model_metrics
from framework.schema import CATEGORIES, GateSettings
from framework.storage import read_results, write_csv
from regression.compare_runs import compare
from regression.release_gate import check_gates


def make_run(model="tiny", run_id="run", per_category=20):
    rows = []
    for category in CATEGORIES:
        for i in range(per_category):
            rows.append(
                dict(
                    run_id=run_id,
                    model=model,
                    test_id=f"{category}_{i}",
                    category=category,
                    passed=True,
                    score=1.0,
                    latency=1.0,
                    status="ok",
                    failure_type="",
                    scoring_hash=f"{category}_{i}",
                    evaluator_version="1",
                    suite="benchmark",
                    run_complete=True,
                    expected_count=len(CATEGORIES) * per_category,
                    prompt="prompt",
                )
            )
    return pd.DataFrame(rows)


def test_new_and_fixed_failures():
    old, new = make_run(), make_run(run_id="new")
    old.loc[0, "passed"] = False
    new.loc[1, "passed"] = False
    report = compare(old, new)["models"][0]
    assert report["new_failures"] == [new.loc[1, "test_id"]]
    assert report["fixed_failures"] == [old.loc[0, "test_id"]]
    assert next(m for m in report["metrics"] if m["metric"] == "overall_pass_rate")["delta"] == 0


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "scoring", "suite", "version", "incomplete", "duplicate"]
)
def test_incompatible_comparisons_rejected(kind):
    old, new = make_run(), make_run()
    if kind == "missing":
        new = new.iloc[1:]
    if kind == "extra":
        new.loc[len(new)] = new.iloc[0].to_dict() | {"test_id": "extra"}
    if kind == "scoring":
        new.loc[0, "scoring_hash"] = "changed"
    if kind == "suite":
        new["suite"] = "adversarial"
    if kind == "version":
        new["evaluator_version"] = "2"
    if kind == "incomplete":
        new["run_complete"] = False
    if kind == "duplicate":
        new.loc[1, "test_id"] = new.loc[0, "test_id"]
    with pytest.raises(ValueError):
        compare(old, new)


def test_model_replacement_requires_mapping():
    old, new = make_run("old"), make_run("new")
    with pytest.raises(ValueError, match="Model sets differ"):
        compare(old, new)
    assert compare(old, new, "old", "new")["models"][0]["tests"] == 160


def test_prompt_changes_are_reported():
    old, new = make_run(), make_run()
    new.loc[0, "prompt"] = "revised prompt"
    assert compare(old, new)["models"][0]["prompt_changes"] == 1


def test_gate_boundary_is_percentage_points():
    old, new = make_run(), make_run()
    new.loc[new.category == "reasoning", "passed"] = [False] + [True] * 19
    assert check_gates(new, GateSettings(), old)["passed"]
    new.loc[new.category == "reasoning", "passed"] = [False] * 2 + [True] * 18
    result = check_gates(new, GateSettings(), old)
    assert not result["passed"]
    assert any("10.0 percentage points" in reason for reason in result["reasons"])


def test_absolute_gates_per_model_and_missing_categories():
    good, bad = make_run("good"), make_run("bad")
    bad.loc[bad.category == "structured_output", "passed"] = False
    assert not check_gates(pd.concat([good, bad]), GateSettings())["passed"]
    subset = good[good.category != "hallucination"].copy()
    subset["expected_count"] = len(subset)
    result = check_gates(subset, GateSettings())
    assert not result["passed"]
    assert any("missing required categories" in reason for reason in result["reasons"])


def test_hallucination_rate_does_not_count_infrastructure_errors_as_hallucinations():
    frame = make_run(per_category=1)
    index = frame[frame.category == "hallucination"].index[0]
    frame.loc[index, ["passed", "status"]] = [False, "generation_error"]
    metrics = model_metrics(frame).iloc[0]
    assert metrics.hallucination_rate is None
    assert metrics.error_rate == 1 / 8
    assert not check_gates(frame, GateSettings())["passed"]


def test_hallucination_threshold():
    frame = make_run()
    indexes = frame[frame.category == "hallucination"].index
    frame.loc[indexes[:2], "passed"] = False
    assert check_gates(frame, GateSettings())["passed"]
    frame.loc[indexes[2], "passed"] = False
    assert not check_gates(frame, GateSettings())["passed"]


def test_result_csv_boolean_validation(tmp_path):
    path = tmp_path / "results.csv"
    frame = make_run()
    frame.loc[0, "passed"] = False
    write_csv(frame, path)
    loaded = read_results(path)
    assert not loaded.passed.iloc[0]
    frame["passed"] = "nope"
    write_csv(frame, path)
    with pytest.raises(ValueError, match="true/false"):
        read_results(path)


def test_smoke_subset_blocks_release():
    frame = make_run(per_category=1)
    frame["dataset_complete"] = False
    result = check_gates(frame, GateSettings())
    assert not result["passed"]
    assert any("smoke-test" in reason for reason in result["reasons"])


def test_auxiliary_errors_block_release():
    frame = make_run()
    frame["judge_error"] = ""
    frame.loc[0, "judge_error"] = "Invalid judge JSON"
    assert not check_gates(frame, GateSettings())["passed"]


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1])
def test_invalid_latency_csv_rejected(tmp_path, value):
    path = tmp_path / "results.csv"
    frame = make_run()
    frame.loc[0, "latency"] = value
    write_csv(frame, path)
    with pytest.raises(ValueError):
        read_results(path)
