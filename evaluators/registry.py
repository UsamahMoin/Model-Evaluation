from evaluators import (
    exact_match,
    format_validator,
    hallucination,
    instruction_following,
    keyword_match,
    semantic_similarity,
)
from framework.schema import Evaluation, EvaluationCase, SemanticSettings

EVALUATOR_VERSION = "1.0"
FAILURES = {
    "factual_accuracy": "FACTUAL_ERROR",
    "instruction_following": "INSTRUCTION_VIOLATION",
    "structured_output": "FORMAT_ERROR",
    "reasoning": "INCORRECT_REASONING",
    "hallucination": "HALLUCINATION",
    "abstention": "UNDER_REFUSAL",
    "context_adherence": "CONTEXT_VIOLATION",
    "robustness": "INSTRUCTION_VIOLATION",
}


def evaluate(
    case: EvaluationCase, response: str, semantic: SemanticSettings | None = None
) -> Evaluation:
    rules, method = case.constraints, case.evaluation_method
    if method == "exact_match":
        result = exact_match.evaluate(response, case.expected_answer, rules.case_sensitive)
    elif method == "keyword_match":
        result = keyword_match.evaluate(
            response, rules.required_keywords, rules.prohibited_text, rules.case_sensitive
        )
    elif method in {"json_format", "schema"}:
        result = format_validator.evaluate(
            response, rules.json_schema, rules.exact_items, rules.required_fields
        )
    elif method == "instruction":
        result = instruction_following.evaluate(response, rules)
    elif method == "hallucination":
        result = hallucination.evaluate(response, case.expected_answer)
    elif method == "semantic":
        settings = semantic or SemanticSettings()
        result = semantic_similarity.evaluate(
            response,
            case.expected_answer,
            model=settings.model,
            threshold=settings.threshold,
            local_files_only=settings.local_files_only,
        )
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
    if not result.passed and not result.failure_type:
        result.failure_type = FAILURES[case.category]
        if "answerable_control" in case.tags:
            result.failure_type = "FACTUAL_ERROR"
    if (
        not result.passed
        and "answerable_control" in case.tags
        and response.strip().casefold()
        in {"unknown", "insufficient context", "ambiguous", "cannot determine"}
    ):
        result.failure_type = "OVER_REFUSAL"
    return result
