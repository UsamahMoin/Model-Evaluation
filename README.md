# Model-Evaluation · Local LLM Lab

Evaluate small local LLMs, inspect failure patterns, and catch regressions before accepting
model or prompt changes. Built for Apple Silicon with Python and Ollama. No NVIDIA GPU,
API key, or paid inference service required.

**Starter models:** `llama3.2:3b` and `qwen2.5:3b`. Candidates run sequentially and are
unloaded between model blocks. Add `gemma3:4b` after establishing the first baseline.

## Quick start on a Mac

Use Python 3.11 or 3.12. Install [Ollama](https://ollama.com/download/mac) and launch it
(or run `ollama serve` in another terminal if no server is running).

```bash
git clone https://github.com/UsamahMoin/Model-Evaluation.git
cd Model-Evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
python -m scripts.validate_dataset
pytest -q
python evaluation_runner.py --baseline
streamlit run dashboard/app.py
```

Open **http://127.0.0.1:8501**. Downloads require internet and several GB of disk space;
evaluation uses local Ollama. A full benchmark is 56 cases × 2 models = **112 generations**.
For a smoke test, use `python evaluation_runner.py --limit 8` (one case per category per
model). Smoke-test subsets cannot establish release readiness.

## Implemented capabilities

| Capability | Implementation |
|---|---|
| Dataset | 56 benchmark cases, 7 per category; 24 adversarial cases, 3 per category |
| Deterministic checks | Exact match, keywords, strict JSON, JSON Schema, instruction constraints, closed-world hallucination probes |
| Metrics | Overall/category pass rates, hallucination probe failure rate, mean/p95 latency, execution errors |
| Failure analysis | Taxonomy, individual prompts/responses, references, scores and explanations |
| Semantic signal | Optional cached Sentence Transformers embeddings and cosine similarity |
| Local judge | Schema-constrained Ollama response, strict Pydantic validation, separate verdicts |
| Judge calibration | Blind 40-response review sheet, agreement, Cohen's kappa, confusion matrix, disagreement export |
| Persistence | SQLite per response, timestamped run directories, CSV exports and run provenance |
| Regression | Matched cases, new/fixed failures, deltas, explicit model replacement mapping |
| Release gates | Absolute quality requirements, category regression limits, coverage/error checks |
| Dashboard | Model comparison, failures, regression, test explorer, judge calibration |
| CI | Offline pytest, Ruff, dataset checks on Python 3.11 and 3.12 |

## Run a regression experiment

```bash
# First baseline only; refuses to overwrite an existing baseline.
python evaluation_runner.py --baseline

# Change the prompt, preserving the cases and scoring rules.
python evaluation_runner.py --system-prompt \
  'Follow the requested format exactly. Return JSON without markdown fences. Check arithmetic before answering. Do not invent missing information.'

python -m regression.compare_runs \
  results/baseline_results.csv results/latest_results.csv \
  --output results/regression_report.json

python -m regression.release_gate results/latest_results.csv \
  --baseline results/baseline_results.csv
```

The runner exits `0` on completion without execution/auxiliary errors, `2` for execution
problems, and `130` on interruption. **Quality failures do not mean the runner failed.**
The release gate exits `0` only when requirements pass, otherwise `1`. Comparison exits
`2` for incompatible runs.

Promote a reviewed result explicitly:

```bash
python -m regression.baseline results/<run-id>/results.csv
# Explicit replacement when a baseline already exists:
python -m regression.baseline results/<run-id>/results.csv --replace
```

To compare different model names, provide both `--baseline-model` and `--current-model`
to the comparison or release-gate command. For example:

```bash
python -m regression.compare_runs results/baseline_results.csv results/latest_results.csv \
  --baseline-model llama3.2:3b --current-model qwen2.5:3b
```

Comparisons require the same case IDs, categories, suite, evaluator version, and scoring
hashes. Changed prompts are allowed and counted. Changed references/schemas/scoring rules
require a new baseline. The program never silently compares only overlapping test IDs.
Bump `EVALUATOR_VERSION` in `evaluators/registry.py` when changing scoring behavior.

### Default release gates

| Requirement | Threshold |
|---|---:|
| Overall pass rate | ≥85% |
| Hallucination probe failure rate | ≤10% |
| Instruction following | ≥90% |
| Structured output | ≥95% |
| Maximum drop in any category | 5 percentage points |

Gates apply **per model** and require all eight categories. Incomplete runs, missing rows,
execution/auxiliary errors and smoke-test subsets block release. Without `--baseline`,
only absolute gates run; the report says regression was not checked. A change from 94%
to 88% is a **6 percentage point** drop.

## Dataset and scoring contracts

Each JSONL case includes `id`, `category`, `prompt`, `expected_answer`, `evaluation_method`,
`difficulty`, `tags`, and optional `constraints`/`rubric`. Example:

```json
{"id":"example_001","category":"structured_output","prompt":"Return exactly three programming languages as a JSON array.","expected_answer":"[\"Python\",\"Java\",\"Rust\"]","evaluation_method":"json_format","difficulty":"easy","tags":["json"],"constraints":{"exact_items":3}}
```

Categories: factual accuracy, instruction following, structured output, reasoning,
hallucination, abstention, context adherence, robustness.

Methods: `exact_match`, `keyword_match`, `json_format`, `schema`, `instruction`,
`hallucination`, `semantic`. References are strings or lists of accepted strings; JSON
references are encoded as strings. Pydantic rejects unknown fields/categories, missing
rules, invalid schemas and duplicate IDs before generation.

- **Exact match:** strips/collapses whitespace and ignores case by default. Punctuation
  and extra prose still fail. Set `case_sensitive: true` when needed; use an accepted
  answer list for legitimate alternative forms.
- **Keywords:** requires whole terms/phrases and rejects prohibited terms. Presence does
  not establish truth; negated statements can still contain the required keywords.
- **JSON:** checks the entire response; fences, trailing prose, duplicate keys, `NaN` and
  `Infinity` fail. No repair hides formatting errors.
- **Schema:** uses JSON Schema Draft 2020-12 for types, required fields, constants, arrays
  and extra fields. Inline schemas only; remote references are rejected.
- **Instructions:** exact bullet counts with no extra lines, whitespace-based word limits,
  required/prohibited terms, JSON fields, schemas and array lengths.
- **Hallucination:** bounded closed-world cases require a specific abstention token for
  absent information. Their failure rate is a **proxy**, not a general hallucination
  detector. A differently phrased refusal can fail strict matching.
- **Abstention:** mixes unanswerable prompts and answerable controls to expose excessive
  refusal. Inspect responses before interpreting any failure label.

Failure types: `HALLUCINATION`, `INSTRUCTION_VIOLATION`, `INCORRECT_REASONING`, `FORMAT_ERROR`,
`CONTEXT_VIOLATION`, `OVER_REFUSAL`, `UNDER_REFUSAL`, `FACTUAL_ERROR`. Infrastructure failures
use `GENERATION_ERROR` or `EVALUATION_ERROR`. Labels describe failed checks, not proven causes.

### Adversarial tests

```bash
python evaluation_runner.py --dataset datasets/adversarial.jsonl
```

Includes conflicting instructions, misleading premises, prompt injection, fake entities,
impossible/missing-context questions, ambiguity, malformed JSON, long context, and prompt
variations. Compare normal/adversarial summaries in the dashboard. Different suites are
not matched regression comparisons. All runs remain available when latest is replaced.

## Semantic evaluation

```bash
pip install -r requirements-semantic.txt
python -m scripts.cache_embeddings         # One explicit download
python evaluation_runner.py --semantic     # Cached files only by default
```

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`, running on CPU to leave
memory for Ollama. Cosine similarity is clipped to 0–1 and stored separately without
changing deterministic verdicts. A case using method `semantic` uses the configured
threshold as its primary score. Similarity is unreliable for numbers, negation and truth;
calibrate thresholds against human labels before using them as quality gates.

## Local judge and human calibration

```bash
python evaluation_runner.py --judge
python -m evaluators.judge_agreement sample results/latest_results.csv --count 40
# Personally fill human_passed and human_reason in results/human_labels.csv.
python -m evaluators.judge_agreement score \
  results/latest_results.csv results/human_labels.csv
```

The default judge is Qwen. Candidate generations finish before judging starts, reducing
model swapping. The judge sees the question, reference and rubric and returns integer
1–5 scores for correctness, relevance and instruction following, plus a reason. Every
dimension must be ≥4 for judge PASS. Invalid/missing output remains an explicit error.
Judge verdicts never overwrite deterministic results. Self-judging introduces bias.

The review sheet hides both automated verdicts. Label 30–50 responses with `PASS`/`FAIL`,
leaving uncertain rows blank. IDs bind labels to the exact run/model/test. Reports include
agreement, a descriptive Wilson interval, Cohen's kappa, confusion matrix, missing-judge
counts and disagreements. Inspect errors for leniency, format blindness, reference
mistakes or injection susceptibility. **Human labels are not fabricated; agreement is
not claimed until a person completes the review.**

## Structure and reproducibility

```text
config.yaml                         Models, prompts, generation settings, gates
datasets/{benchmark,adversarial}.jsonl
models/ollama_client.py              generate(model, prompt), timing, unload
evaluators/                         Deterministic checks, semantic, judge, calibration
framework/                          Pydantic contracts, metrics, SQLite/CSV
runner/evaluation_runner.py          Sequential execution and provenance
regression/                         Comparison, release gate, baseline promotion
dashboard/app.py                    Streamlit application
tests/                              Offline evaluator, runner, gate and dashboard tests
results/
  evaluations.sqlite
  baseline_results.csv
  latest_results.csv
  <run-id>/{results.csv,summary.csv,metadata.json}
```

Metadata includes configuration/hash, selected dataset hash, Ollama model digests,
evaluator/judge versions and Git commit. Rows include prompts, references, responses,
latency, load duration, token count, stop reason and scoring hash. Temperature 0/seed 42
reduce variability but do not guarantee identical results across hardware/runtime/model
revisions. Repeat trials before interpreting small deltas.

Overall pass rate includes execution errors. Hallucination rate uses only successfully
evaluated hallucination probes; no valid probes makes it undefined and fails the gate.
Generation latency includes loading; judge/semantic time is excluded. Seven cases per
category yields coarse 14.3-point steps: expand to 150–300 reviewed cases for stronger evidence.

Results and human labels are ignored by Git. SQLite commits each response. Interruptions
preserve saved rows without replacing latest/baseline. The dashboard binds to loopback,
Streamlit telemetry is disabled, and Ollama URLs are restricted to loopback. Generated
code and URLs are never executed or followed.

## Development and troubleshooting

```bash
ruff check .
pytest -q
python -m compileall -q framework models evaluators runner regression dashboard scripts
```

- Connection refused: start Ollama and check `ollama list`.
- Missing model: pull the exact tag in `config.yaml`.
- Memory pressure: close other models/apps, reduce `num_ctx`/`num_predict`, and select one
  candidate with `--models llama3.2:3b`. Avoid concurrent model loading.
- Missing embeddings: install optional requirements and run the cache script once.
- Generation errors: inspect row status/reason, fix the runtime, then rerun.
- Release failed: inspect category deltas/new failures; do not silently lower thresholds.
- Third model: `ollama pull gemma3:4b`, add it to `config.yaml`, then create a reviewed
  baseline including it or use explicit model mapping.

GitHub Actions uses deterministic fixtures/mocked Ollama and needs no GPU or model
server. CI validates the framework; it does not claim live model quality.

## Primary implementation references

- [Ollama generate API](https://docs.ollama.com/api/generate)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Sentence Transformers similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
