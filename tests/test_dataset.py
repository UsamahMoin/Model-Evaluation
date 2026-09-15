from collections import Counter

import pytest
from pydantic import ValidationError

from framework.schema import (
    CATEGORIES,
    Config,
    Constraints,
    EvaluationCase,
    OllamaSettings,
    load_config,
    load_dataset,
)


def test_dataset_size_balance_and_ids():
    for path, count in [("datasets/benchmark.jsonl", 56), ("datasets/adversarial.jsonl", 24)]:
        cases = load_dataset(path)
        assert len(cases) == count
        counts = Counter(case.category for case in cases)
        assert set(counts) == set(CATEGORIES)
        assert len(set(counts.values())) == 1
        assert len({case.id for case in cases}) == count


def test_bad_dataset_fails_with_line_number(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id":"wrong"}\n')
    with pytest.raises(ValueError, match=":1:"):
        load_dataset(path)
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)


def test_duplicate_cases_rejected(tmp_path):
    case = load_dataset("datasets/benchmark.jsonl")[0]
    path = tmp_path / "cases.jsonl"
    path.write_text(case.model_dump_json() + "\n" + case.model_dump_json())
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


@pytest.mark.parametrize(
    "method,rules",
    [("exact_match", {}), ("schema", {}), ("keyword_match", {}), ("instruction", {})],
)
def test_missing_evaluation_rules(method, rules):
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="a",
            category="reasoning",
            prompt="A?",
            difficulty="easy",
            evaluation_method=method,
            constraints=rules,
        )


def test_remote_schema_references_rejected():
    with pytest.raises(ValidationError):
        Constraints(json_schema={"$ref": "https://example.com/schema"})


def test_local_config():
    assert len(load_config("config.yaml").models) == 2
    with pytest.raises(ValidationError):
        OllamaSettings(base_url="https://external.example")
    with pytest.raises(ValidationError):
        Config(models=["x"], prompt_template="{prompt.__class__}")
    with pytest.raises(ValidationError):
        Config(models=["x", "x"])
