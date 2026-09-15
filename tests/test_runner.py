import argparse
import json

import httpx
import pytest
import yaml

from framework.schema import load_dataset
from framework.storage import Store, read_results
from models.ollama_client import Generation, OllamaClient
from runner.evaluation_runner import balanced_limit, run


def test_ollama_request_and_metadata():
    def respond(request):
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["prompt"] == "hello"
        assert body["model"] == "tiny"
        assert body["options"]["seed"] == 42
        return httpx.Response(
            200,
            json={
                "response": "world",
                "done": True,
                "load_duration": 1_000_000_000,
                "eval_count": 4,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(respond))
    result = client.generate("tiny", "hello", options={"seed": 42})
    client.close()
    assert result.response == "world"
    assert result.latency >= 0
    assert result.load_duration == 1


def test_invalid_ollama_completion():
    client = OllamaClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"response": "partial", "done": False})
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        client.generate("tiny", "hello")
    client.close()


def test_balanced_smoke_limit():
    cases = balanced_limit(load_dataset("datasets/benchmark.jsonl"), 8)
    assert len({case.category for case in cases}) == 8


class FakeClient:
    def __init__(self, failure=None):
        self.failure = failure

    def models(self):
        return {"tiny": "digest"}

    def generate(self, model, prompt, **kwargs):
        if self.failure:
            raise self.failure
        return Generation(model, prompt, "Paris", 0.2)

    def unload(self, model):
        pass

    def close(self):
        pass


def args_for(tmp_path):
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(load_dataset("datasets/benchmark.jsonl")[0].model_dump_json() + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(dict(models=["tiny"], dataset=str(dataset), results_dir="results"))
    )
    return argparse.Namespace(
        config=str(config),
        models=None,
        dataset=None,
        limit=None,
        judge=False,
        semantic=False,
        baseline=False,
    )


def test_runner_records_and_exports(tmp_path):
    args = args_for(tmp_path)
    assert run(args, FakeClient()) == 0
    frame = read_results(tmp_path / "results/latest_results.csv")
    assert frame.passed.all() and frame.run_complete.all()
    store = Store(tmp_path / "results/evaluations.sqlite")
    assert store.frame(frame.run_id.iloc[0]).run_complete.all()
    store.close()


def test_generation_error_persisted_without_baseline(tmp_path):
    args = args_for(tmp_path)
    args.baseline = True
    assert run(args, FakeClient(RuntimeError("timeout"))) == 2
    frame = read_results(tmp_path / "results/latest_results.csv")
    assert frame.status.iloc[0] == "generation_error"
    assert frame.failure_type.iloc[0] == "GENERATION_ERROR"
    assert not (tmp_path / "results/baseline_results.csv").exists()


def test_interrupted_run_preserves_latest(tmp_path):
    args = args_for(tmp_path)
    run(args, FakeClient())
    latest = tmp_path / "results/latest_results.csv"
    original = latest.read_bytes()
    with pytest.raises(KeyboardInterrupt):
        run(args, FakeClient(KeyboardInterrupt()))
    assert latest.read_bytes() == original
    metadata = [
        json.loads(path.read_text()) for path in (tmp_path / "results").glob("*/metadata.json")
    ]
    assert any(run["status"] == "interrupted" for run in metadata)
