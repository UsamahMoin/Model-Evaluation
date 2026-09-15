import pandas as pd
import pytest

from evaluators.judge_agreement import agreement, review_sample


def sample_results():
    return pd.DataFrame(
        [
            dict(
                run_id="r",
                model="m",
                test_id=str(i),
                category="factual_accuracy",
                prompt="Q",
                expected_answer="A",
                response="A",
                status="ok",
                judge_passed=judge,
            )
            for i, judge in enumerate([True, True, False, False])
        ]
    )


def test_blind_review_sheet_has_no_verdicts():
    sample = review_sample(sample_results())
    assert "judge_passed" not in sample and "passed" not in sample
    assert sample.human_passed.eq("").all()


def test_agreement_and_disagreements():
    results = sample_results()
    labels = results[["run_id", "model", "test_id"]].copy()
    labels["human_passed"] = ["PASS", "FAIL", "FAIL", "FAIL"]
    report, disagreements = agreement(results, labels)
    assert report["agreement"] == 0.75
    assert report["compared"] == 4
    assert report["confusion_matrix_human_rows_judge_columns"] == [[2, 1], [0, 1]]
    assert disagreements.test_id.tolist() == ["1"]


def test_missing_judge_reported_and_blank_labels_excluded():
    results = sample_results()
    results["judge_passed"] = results.judge_passed.astype(object)
    results.loc[0, "judge_passed"] = None
    labels = results[["run_id", "model", "test_id"]].copy()
    labels["human_passed"] = ["PASS", "FAIL", "FAIL", ""]
    report, _ = agreement(results, labels)
    assert report["excluded_missing_judge"] == 1
    assert report["labeled"] == 3
    assert report["compared"] == 2


def test_foreign_labels_and_invalid_labels_rejected():
    results = sample_results()
    labels = review_sample(results)
    with pytest.raises(ValueError, match="No completed"):
        agreement(results, labels)
    labels["human_passed"] = "maybe"
    with pytest.raises(ValueError, match="PASS/FAIL"):
        agreement(results, labels)
    labels["human_passed"] = "PASS"
    labels.loc[0, "run_id"] = "different"
    with pytest.raises(ValueError, match="absent"):
        agreement(results, labels)
