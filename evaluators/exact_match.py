import re

from framework.schema import Evaluation


def normalize(text: str, case_sensitive: bool = False) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text if case_sensitive else text.casefold()


def evaluate(response: str, expected: str | list[str], case_sensitive: bool = False) -> Evaluation:
    answers = [expected] if isinstance(expected, str) else expected
    passed = normalize(response, case_sensitive) in {
        normalize(answer, case_sensitive) for answer in answers
    }
    return Evaluation(
        passed=passed,
        score=float(passed),
        reason="" if passed else f"Expected one of {answers!r}; received {response!r}.",
    )
