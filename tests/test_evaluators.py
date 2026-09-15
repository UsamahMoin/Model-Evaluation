import json

import pytest
from pydantic import ValidationError

from evaluators import (
    exact_match,
    format_validator,
    hallucination,
    instruction_following,
    keyword_match,
)
from evaluators.llm_judge import JudgeVerdict
from evaluators.registry import evaluate
from framework.schema import Constraints, EvaluationCase, load_dataset


@pytest.mark.parametrize(
    "response,expected,passed",
    [
        ("  PARIS\n", "Paris", True),
        ("New   York", "New York", True),
        ("Paris.", "Paris", False),
        ("The answer is Paris", "Paris", False),
        ("Pacific", ["Pacific", "Pacific Ocean"], True),
        ("", "Paris", False),
    ],
)
def test_exact_match(response, expected, passed):
    assert exact_match.evaluate(response, expected).passed is passed


def test_exact_case_sensitive():
    assert not exact_match.evaluate("au", "Au", True).passed


def test_keywords_boundaries_and_prohibitions():
    assert keyword_match.evaluate("Plants absorb carbon\n dioxide.", ["carbon dioxide"]).passed
    assert not keyword_match.evaluate("concatenate", ["cat"]).passed
    assert not keyword_match.evaluate("cat and motor", ["cat"], ["motor"]).passed
    assert not keyword_match.evaluate("cat", ["Cat"], case_sensitive=True).passed


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"ok":true}\n```',
        '{"a":1,}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1,"a":2}',
        '{"nested":{"x":1,"x":2}}',
        "",
        "{} trailing",
    ],
)
def test_strict_json_rejects_invalid_output(response):
    assert not format_validator.evaluate(response).passed


@pytest.mark.parametrize("response", ["null", "[]", '{"ok":true}', '"text"', "42"])
def test_valid_json_types(response):
    assert format_validator.evaluate(response).passed


def test_schema_types_and_extra_fields():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }
    assert format_validator.evaluate('{"count":3}', schema).passed
    for response in ['{"count":"3"}', '{"count":true}', "{}", '{"count":3,"admin":true}']:
        assert not format_validator.evaluate(response, schema).passed


def test_json_item_counts_and_fields():
    assert format_validator.evaluate("[]", exact_items=0).passed
    assert not format_validator.evaluate("{}", exact_items=0).passed
    assert not format_validator.evaluate("[1,2,3,4,5]", exact_items=3).passed
    assert not format_validator.evaluate("[]", required_fields=["name"]).passed


def test_instruction_bullets_and_extra_text():
    rules = Constraints(exact_bullets=3)
    assert instruction_following.evaluate("- a\n- b\n- c", rules).passed
    for response in [
        "Heading\n- a\n- b\n- c",
        "- a\n- b",
        "- a\n- b\n- c\n- d",
        "1. a\n2. b\n3. c",
    ]:
        assert not instruction_following.evaluate(response, rules).passed


def test_combined_instruction_rules():
    rules = Constraints(max_words=4, required_keywords=["river"], prohibited_text=["ocean"])
    assert instruction_following.evaluate("The river flows.", rules).passed
    assert not instruction_following.evaluate("The river flows into an ocean.", rules).passed
    assert not instruction_following.evaluate("", Constraints(max_words=100)).passed


def test_hallucination_is_closed_world_check():
    assert hallucination.evaluate("UNKNOWN", "UNKNOWN").passed
    assert not hallucination.evaluate("The answer is 2050", "UNKNOWN").passed


def test_over_refusal_control():
    case = EvaluationCase(
        id="control",
        category="abstention",
        prompt="What is 2+2?",
        expected_answer="4",
        evaluation_method="exact_match",
        difficulty="easy",
        tags=["answerable_control"],
    )
    assert evaluate(case, "UNKNOWN").failure_type == "OVER_REFUSAL"
    assert evaluate(case, "5").failure_type == "FACTUAL_ERROR"


@pytest.mark.parametrize(
    "update",
    [{"correctness": 6}, {"relevance": "5"}, {"correctness": True}, {"reason": ""}, {"extra": 1}],
)
def test_judge_schema_rejects_bad_scores(update):
    verdict = dict(correctness=5, relevance=5, instruction_following=5, reason="Correct")
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(verdict | update)


def test_judge_pass_uses_all_dimensions():
    verdict = JudgeVerdict(
        correctness=5, relevance=5, instruction_following=2, reason="Wrong format"
    )
    assert not verdict.passed()


@pytest.mark.parametrize("path", ["datasets/benchmark.jsonl", "datasets/adversarial.jsonl"])
def test_every_reference_satisfies_its_own_rubric(path):
    for case in load_dataset(path):
        reference = (
            case.expected_answer[0]
            if isinstance(case.expected_answer, list)
            else case.expected_answer
        )
        result = evaluate(case, reference)
        assert result.passed, f"{case.id}: {result.reason}; {json.dumps(reference)}"
