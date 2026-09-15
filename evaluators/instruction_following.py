import re

from evaluators import format_validator, keyword_match
from framework.schema import Constraints, Evaluation


def evaluate(response: str, rules: Constraints) -> Evaluation:
    reasons = []
    if not response.strip():
        reasons.append("Response is empty")
    if rules.exact_bullets is not None:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        bullets = [line for line in lines if re.match(r"^[-*•]\s+\S", line)]
        if len(bullets) != rules.exact_bullets or len(lines) != len(bullets):
            reasons.append(
                f"Expected exactly {rules.exact_bullets} bullet lines and no extra text; received {len(bullets)} bullets in {len(lines)} lines"
            )
    if rules.max_words is not None and len(response.split()) > rules.max_words:
        reasons.append(
            f"Expected at most {rules.max_words} words; received {len(response.split())}"
        )
    words = keyword_match.evaluate(
        response, rules.required_keywords, rules.prohibited_text, rules.case_sensitive
    )
    if not words.passed:
        reasons.append(words.reason)
    if rules.json_schema is not None or rules.exact_items is not None or rules.required_fields:
        formatted = format_validator.evaluate(
            response, rules.json_schema, rules.exact_items, rules.required_fields
        )
        if not formatted.passed:
            reasons.append(formatted.reason)
    return Evaluation(
        passed=not reasons,
        score=float(not reasons),
        reason="; ".join(reasons),
        failure_type="INSTRUCTION_VIOLATION" if reasons else "",
    )
