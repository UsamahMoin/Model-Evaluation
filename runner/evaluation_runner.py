from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from evaluators import llm_judge, semantic_similarity
from evaluators.registry import EVALUATOR_VERSION, evaluate
from framework.metrics import model_metrics
from framework.schema import CATEGORIES, Evaluation, fingerprint, load_config, load_dataset
from framework.storage import Store, write_csv
from models.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[1]


def balanced_limit(cases, limit):
    if limit is None:
        return cases
    buckets = [[case for case in cases if case.category == category] for category in CATEGORIES]
    ordered = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                ordered.append(bucket.pop(0))
    return ordered[:limit]


def run(args, client=None):
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base = config_path.parent
    if args.models:
        config.models = list(dict.fromkeys(args.models))
    if args.judge:
        config.judge.enabled = True
    if args.semantic:
        config.semantic.enabled = True
    if getattr(args, "system_prompt", None) is not None:
        config.system_prompt = args.system_prompt
    dataset_path = Path(args.dataset) if args.dataset else base / config.dataset
    full_dataset = load_dataset(dataset_path)
    cases = balanced_limit(full_dataset, args.limit)
    if args.baseline and len(cases) < len(full_dataset):
        raise ValueError("Use the full dataset to establish a baseline; remove --limit")
    output_dir = base / config.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline and (output_dir / "baseline_results.csv").exists():
        raise ValueError(
            "Baseline already exists. Use the baseline command to explicitly replace it."
        )
    client = client or OllamaClient(config.ollama)
    try:
        installed = client.models()
        needed = set(config.models) | ({config.judge.model} if config.judge.enabled else set())
        missing = needed - installed.keys()
        if missing:
            raise ValueError(
                "Missing local models. Run: "
                + " && ".join(f"ollama pull {model}" for model in sorted(missing))
            )
        if config.semantic.enabled or any(case.evaluation_method == "semantic" for case in cases):
            semantic_similarity.load_encoder(
                config.semantic.model, config.semantic.local_files_only
            )
        created_at = datetime.now(timezone.utc).isoformat()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = output_dir / run_id
        run_dir.mkdir()
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        git_status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        )
        source_files = [
            path
            for folder in ("models", "evaluators", "framework", "runner", "regression")
            for path in (ROOT / folder).glob("*.py")
        ]
        metadata = {
            "run_id": run_id,
            "created_at": created_at,
            "config": config.model_dump(),
            "config_hash": fingerprint(config.model_dump()),
            "git_commit": git.stdout.strip(),
            "git_dirty": bool(git_status.stdout.strip()),
            "source_hash": fingerprint(
                {str(path.relative_to(ROOT)): path.read_text() for path in sorted(source_files)}
            ),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: version(package)
                for package in ("pydantic", "pandas", "httpx", "jsonschema")
            },
            "dataset": str(dataset_path.resolve()),
            "dataset_hash": fingerprint([c.model_dump() for c in cases]),
            "model_digests": {model: installed[model] for model in needed},
            "evaluator_version": EVALUATOR_VERSION,
            "judge_version": llm_judge.JUDGE_VERSION,
            "expected_count": len(cases) * len(config.models),
            "case_count_per_model": len(cases),
            "status": "running",
            "partial_dataset": len(cases) < len(full_dataset),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        store = Store(output_dir / "evaluations.sqlite")
        store.start(run_id, created_at, metadata)
        status = "interrupted"
        active_model = None
        rows = []
        try:
            # Run each candidate in a block; unload it before loading another on 8 GB Macs.
            for model in config.models:
                active_model = model
                for case in cases:
                    prompt = config.prompt_template.format(prompt=case.prompt)
                    row = {
                        "run_id": run_id,
                        "created_at": created_at,
                        "test_id": case.id,
                        "model": model,
                        "model_digest": installed[model],
                        "category": case.category,
                        "suite": dataset_path.stem,
                        "prompt": prompt,
                        "original_prompt": case.prompt,
                        "expected_answer": json.dumps(case.expected_answer, ensure_ascii=False),
                        "evaluation_method": case.evaluation_method,
                        "difficulty": case.difficulty,
                        "tags": json.dumps(case.tags),
                        "scoring_hash": fingerprint(
                            {
                                "case": case.scoring_hash,
                                "semantic": config.semantic.model_dump()
                                if case.evaluation_method == "semantic"
                                else None,
                            }
                        ),
                        "evaluator_version": EVALUATOR_VERSION,
                        "config_hash": metadata["config_hash"],
                        "dataset_hash": metadata["dataset_hash"],
                        "dataset_complete": len(cases) == len(full_dataset),
                        "status": "ok",
                        "response": "",
                        "latency": 0.0,
                        "load_duration": 0.0,
                        "eval_count": 0,
                        "done_reason": "",
                        "semantic_score": None,
                        "semantic_error": "",
                        "judge_json": "",
                        "judge_passed": None,
                        "judge_error": "",
                        "judge_model": config.judge.model if config.judge.enabled else "",
                        "judge_threshold": config.judge.pass_threshold
                        if config.judge.enabled
                        else None,
                        "judge_version": llm_judge.JUDGE_VERSION if config.judge.enabled else "",
                        "expected_count": len(cases),
                        "run_complete": False,
                    }
                    started = time.perf_counter()
                    try:
                        generated = client.generate(
                            model,
                            prompt,
                            system=config.system_prompt,
                            options=config.generation.model_dump(),
                        )
                        for field in (
                            "response",
                            "latency",
                            "load_duration",
                            "eval_count",
                            "done_reason",
                        ):
                            row[field] = getattr(generated, field)
                    except Exception as exc:
                        row["status"] = "generation_error"
                        row["latency"] = time.perf_counter() - started
                        outcome = Evaluation(
                            passed=False, score=0, reason=str(exc), failure_type="GENERATION_ERROR"
                        )
                    else:
                        try:
                            outcome = evaluate(case, row["response"], config.semantic)
                        except Exception as exc:
                            row["status"] = "evaluation_error"
                            outcome = Evaluation(
                                passed=False,
                                score=0,
                                reason=str(exc),
                                failure_type="EVALUATION_ERROR",
                            )
                        if config.semantic.enabled and case.expected_answer:
                            try:
                                signal = semantic_similarity.evaluate(
                                    row["response"],
                                    case.expected_answer,
                                    model=config.semantic.model,
                                    threshold=config.semantic.threshold,
                                    local_files_only=config.semantic.local_files_only,
                                )
                                row["semantic_score"] = signal.score
                            except Exception as exc:
                                row["semantic_error"] = str(exc)
                    row.update(
                        score=outcome.score,
                        passed=outcome.passed,
                        failure_reason=outcome.reason,
                        failure_type=outcome.failure_type,
                    )
                    rows.append(row)
                    store.save(row)
                    print(
                        f"[{len(rows)}/{metadata['expected_count']}] {model} {case.id}: {'PASS' if outcome.passed else 'FAIL'} ({row['latency']:.2f}s)",
                        flush=True,
                    )
                client.unload(model)
                active_model = None
            # Judge in a separate pass so candidate and judge models are not alternated per prompt.
            if config.judge.enabled:
                active_model = config.judge.model
                by_id = {case.id: case for case in cases}
                for number, row in enumerate(rows, 1):
                    if row["status"] != "ok":
                        continue
                    try:
                        case = by_id[row["test_id"]].model_copy(update={"prompt": row["prompt"]})
                        verdict = llm_judge.evaluate(
                            client, config.judge.model, case, row["response"]
                        )
                        row["judge_json"] = verdict.model_dump_json()
                        row["judge_passed"] = verdict.passed(config.judge.pass_threshold)
                    except Exception as exc:
                        row["judge_error"] = str(exc)
                    with store.connection:
                        store.connection.execute(
                            "UPDATE results SET payload = ? WHERE run_id = ? AND model = ? AND test_id = ?",
                            (json.dumps(row), run_id, row["model"], row["test_id"]),
                        )
                    print(f"Judge [{number}/{len(rows)}] {row['test_id']}", flush=True)
            status = (
                "completed_with_errors"
                if any(
                    row["status"] != "ok" or row["judge_error"] or row["semantic_error"]
                    for row in rows
                )
                else "completed"
            )
        finally:
            if active_model:
                try:
                    client.unload(active_model)
                except Exception:
                    pass
            metadata["status"] = status
            metadata["recorded_count"] = len(rows)
            store.finish(run_id, status, metadata)
            frame = store.frame(run_id)
            complete = (
                status in {"completed", "completed_with_errors"}
                and len(frame) == metadata["expected_count"]
            )
            if not frame.empty:
                frame["run_complete"] = complete
                # Keep SQLite payloads and CSV exports consistent after finalizing a run.
                with store.connection:
                    for row in frame.to_dict("records"):
                        store.connection.execute(
                            "UPDATE results SET payload = ? WHERE run_id = ? AND model = ? AND test_id = ?",
                            (json.dumps(row), run_id, row["model"], row["test_id"]),
                        )
                write_csv(frame, run_dir / "results.csv")
                model_metrics(frame).to_csv(run_dir / "summary.csv", index=False)
                if complete:
                    write_csv(frame, output_dir / "latest_results.csv")
                    if args.baseline and status == "completed":
                        write_csv(frame, output_dir / "baseline_results.csv")
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            store.close()
        print(f"\nSaved {run_dir / 'results.csv'}\n")
        print(model_metrics(frame).to_string(index=False))
        return 0 if status == "completed" else 2
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate local Ollama models; no cloud services.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--dataset")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--limit", type=int, help="Balanced subset per model for a smoke test")
    parser.add_argument(
        "--baseline", action="store_true", help="Save a first baseline; refuses to overwrite"
    )
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument(
        "--system-prompt", help="Override the configured system prompt for a regression experiment"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        return run(args)
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed rows remain in SQLite and the run directory.", file=sys.stderr
        )
        return 130
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
