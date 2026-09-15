from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import Formatter
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

CATEGORIES = (
    "factual_accuracy",
    "instruction_following",
    "structured_output",
    "reasoning",
    "hallucination",
    "abstention",
    "context_adherence",
    "robustness",
)
Category = Literal[
    "factual_accuracy",
    "instruction_following",
    "structured_output",
    "reasoning",
    "hallucination",
    "abstention",
    "context_adherence",
    "robustness",
]
Method = Literal[
    "exact_match",
    "keyword_match",
    "json_format",
    "schema",
    "instruction",
    "hallucination",
    "semantic",
]


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Constraints(StrictModel):
    case_sensitive: bool = False
    required_keywords: list[str] = Field(default_factory=list)
    prohibited_text: list[str] = Field(default_factory=list)
    exact_bullets: int | None = Field(default=None, ge=1)
    max_words: int | None = Field(default=None, ge=1)
    exact_items: int | None = Field(default=None, ge=0)
    required_fields: list[str] = Field(default_factory=list)
    json_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def valid_schema(self):
        if self.json_schema is not None:
            Draft202012Validator.check_schema(self.json_schema)
            # Keep dataset validation entirely local; remote references are not supported.
            if '"$ref"' in json.dumps(self.json_schema):
                raise ValueError("JSON schema references are not supported; inline the schema")
        for word in self.required_keywords + self.prohibited_text + self.required_fields:
            if not word.strip():
                raise ValueError("Constraint strings must not be blank")
        return self


class EvaluationCase(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    category: Category
    prompt: str = Field(min_length=1)
    expected_answer: str | list[str] | None = None
    evaluation_method: Method
    difficulty: Literal["easy", "medium", "hard"]
    tags: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    rubric: str = "Assess correctness, relevance, and compliance with the explicit request."

    @model_validator(mode="after")
    def sufficient_rules(self):
        if self.evaluation_method in {"exact_match", "hallucination", "semantic"}:
            if not self.expected_answer or (
                isinstance(self.expected_answer, list) and not all(self.expected_answer)
            ):
                raise ValueError("This evaluator requires a nonempty expected_answer")
        if self.evaluation_method == "keyword_match" and not self.constraints.required_keywords:
            raise ValueError("Keyword matching requires required_keywords")
        if self.evaluation_method == "schema" and self.constraints.json_schema is None:
            raise ValueError("Schema evaluation requires json_schema")
        if self.evaluation_method == "instruction" and not any(
            value is not None and value != [] and value is not False
            for key, value in self.constraints.model_dump().items()
            if key != "case_sensitive"
        ):
            raise ValueError("Instruction evaluation requires at least one rule")
        return self

    @property
    def scoring_hash(self) -> str:
        rules = self.model_dump(exclude={"prompt", "tags", "difficulty"})
        return fingerprint(rules)


class OllamaSettings(StrictModel):
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=180, gt=0)
    keep_alive: str = "5m"

    @model_validator(mode="after")
    def local_only(self):
        url = urlparse(self.base_url)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama must use an HTTP loopback address for local-only evaluation")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid Ollama base URL")
        return self


class GenerationSettings(StrictModel):
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 42
    num_ctx: int = Field(default=4096, ge=512)
    num_predict: int = Field(default=256, ge=1)


class SemanticSettings(StrictModel):
    enabled: bool = False
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = True
    threshold: float = Field(default=0.8, ge=0, le=1)


class JudgeSettings(StrictModel):
    enabled: bool = False
    model: str = "qwen2.5:3b"
    pass_threshold: int = Field(default=4, ge=1, le=5)


class GateSettings(StrictModel):
    overall_pass_rate_min: float = Field(default=0.85, ge=0, le=1)
    hallucination_rate_max: float = Field(default=0.10, ge=0, le=1)
    instruction_following_min: float = Field(default=0.90, ge=0, le=1)
    structured_output_min: float = Field(default=0.95, ge=0, le=1)
    max_category_drop: float = Field(default=0.05, ge=0, le=1)


class Config(StrictModel):
    models: list[str] = Field(min_length=1)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    system_prompt: str = ""
    prompt_template: str = "{prompt}"
    dataset: str = "datasets/benchmark.jsonl"
    results_dir: str = "results"
    semantic: SemanticSettings = Field(default_factory=SemanticSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    release_gates: GateSettings = Field(default_factory=GateSettings)

    @model_validator(mode="after")
    def validate_config(self):
        if len(set(self.models)) != len(self.models) or not all(x.strip() for x in self.models):
            raise ValueError("Model names must be nonempty and unique")
        fields = [
            field for _, field, _, _ in Formatter().parse(self.prompt_template) if field is not None
        ]
        if not fields or any(field != "prompt" for field in fields):
            raise ValueError("prompt_template must contain {prompt} and no other placeholders")
        self.prompt_template.format(prompt="test")
        return self


def load_config(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))


def load_dataset(path: str | Path) -> list[EvaluationCase]:
    cases, seen = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = EvaluationCase.model_validate_json(line)
            if case.id in seen:
                raise ValueError(f"Duplicate case ID: {case.id}")
        except Exception as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        cases.append(case)
        seen.add(case.id)
    if not cases:
        raise ValueError(f"Dataset is empty: {path}")
    return cases


class Evaluation(StrictModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    reason: str = ""
    failure_type: str = ""
