from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import framework.schema as schema
from framework.storage import write_csv
from runner.evaluation_runner import balanced_limit


def test_dashboard_loads_and_navigation(tmp_path, monkeypatch):
    config = schema.load_config("config.yaml").model_copy(update={"results_dir": str(tmp_path)})
    monkeypatch.setattr(schema, "load_config", lambda path: config)
    cases = balanced_limit(schema.load_dataset("datasets/benchmark.jsonl"), 8)
    rows = []
    for case in cases:
        rows.append(
            dict(
                run_id="fixture",
                model="fixture-model",
                test_id=case.id,
                category=case.category,
                passed=True,
                score=1.0,
                latency=0.1,
                status="ok",
                failure_type="",
                failure_reason="",
                scoring_hash=case.scoring_hash,
                evaluator_version="1",
                suite="benchmark",
                run_complete=True,
                expected_count=8,
                prompt=case.prompt,
                expected_answer=str(case.expected_answer),
                response="fixture response",
                evaluation_method=case.evaluation_method,
            )
        )
    rows[0].update(
        passed=False, score=0.0, failure_type="FACTUAL_ERROR", failure_reason="Fixture failure"
    )
    frame = pd.DataFrame(rows)
    write_csv(frame, tmp_path / "latest_results.csv")
    write_csv(frame.assign(run_id="baseline"), tmp_path / "baseline_results.csv")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/app.py").run(
        timeout=30
    )
    assert not app.exception
    assert app.metric[0].value == "87.5%"
    for page in [
        "Failure analysis",
        "Regression analysis",
        "Test explorer",
        "Judge calibration",
        "Model comparison",
    ]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, page
    app.sidebar.radio[0].set_value("Test explorer").run(timeout=30)
    next(element for element in app.selectbox if element.label == "Outcome").set_value("FAIL").run(
        timeout=30
    )
    assert not app.exception
    assert len(app.dataframe[0].value) == 1


def test_dashboard_empty_state(tmp_path, monkeypatch):
    config = schema.load_config("config.yaml").model_copy(update={"results_dir": str(tmp_path)})
    monkeypatch.setattr(schema, "load_config", lambda path: config)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/app.py").run(
        timeout=30
    )
    assert not app.exception
    assert "No evaluation runs yet" in app.info[0].value
