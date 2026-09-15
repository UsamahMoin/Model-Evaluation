from evaluators.exact_match import evaluate as exact_match
from framework.schema import Evaluation


def evaluate(response: str, expected: str | list[str]) -> Evaluation:
    """Closed-world probe: require the specified abstention/correction token.

    A failing response is a probe failure, not proof of an invented factual claim.
    This evaluator intentionally does not infer truth from arbitrary prose.
    """
    result = exact_match(response, expected)
    if not result.passed:
        result.reason = "Hallucination probe failed: " + result.reason
        result.failure_type = "HALLUCINATION"
    return result
