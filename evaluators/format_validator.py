import json

from jsonschema import Draft202012Validator

from framework.schema import Evaluation


def parse_json(response: str):
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(response, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def evaluate(
    response: str,
    schema: dict | None = None,
    exact_items: int | None = None,
    required_fields: list[str] | None = None,
) -> Evaluation:
    try:
        value = parse_json(response)
    except (ValueError, TypeError) as exc:
        return Evaluation(
            passed=False, score=0, reason=f"Invalid JSON: {exc}", failure_type="FORMAT_ERROR"
        )
    if schema is not None:
        errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
        if errors:
            return Evaluation(
                passed=False,
                score=0,
                reason="; ".join(error.message for error in errors),
                failure_type="FORMAT_ERROR",
            )
    if exact_items is not None and (not isinstance(value, list) or len(value) != exact_items):
        actual = len(value) if isinstance(value, list) else type(value).__name__
        return Evaluation(
            passed=False,
            score=0,
            reason=f"Expected exactly {exact_items} array items; received {actual}.",
            failure_type="FORMAT_ERROR",
        )
    if required_fields and (
        not isinstance(value, dict) or not set(required_fields) <= value.keys()
    ):
        return Evaluation(
            passed=False,
            score=0,
            reason=f"Expected object containing fields {required_fields}.",
            failure_type="FORMAT_ERROR",
        )
    return Evaluation(passed=True, score=1)
