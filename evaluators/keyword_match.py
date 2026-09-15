import re

from framework.schema import Evaluation


def contains(text: str, phrase: str, case_sensitive: bool = False) -> bool:
    # Word boundaries prevent 'cat' from matching 'concatenate'. Multiword phrases allow whitespace.
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"(?!\w)"
    return re.search(pattern, text, 0 if case_sensitive else re.IGNORECASE) is not None


def evaluate(
    response: str,
    required: list[str],
    prohibited: list[str] | None = None,
    case_sensitive: bool = False,
) -> Evaluation:
    missing = [word for word in required if not contains(response, word, case_sensitive)]
    forbidden = [word for word in prohibited or [] if contains(response, word, case_sensitive)]
    reasons = []
    if missing:
        reasons.append(f"Missing required terms: {', '.join(missing)}")
    if forbidden:
        reasons.append(f"Contains prohibited terms: {', '.join(forbidden)}")
    passed = not reasons
    return Evaluation(passed=passed, score=float(passed), reason="; ".join(reasons))
