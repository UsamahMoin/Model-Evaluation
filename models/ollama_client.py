from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from framework.schema import OllamaSettings


@dataclass
class Generation:
    model: str
    prompt: str
    response: str
    latency: float
    load_duration: float = 0
    eval_count: int = 0
    done_reason: str = ""


class OllamaClient:
    def __init__(self, settings: OllamaSettings | None = None, transport=None):
        self.settings = settings or OllamaSettings()
        self.http = httpx.Client(
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    def close(self):
        self.http.close()

    def models(self) -> dict[str, str]:
        response = self.http.get("/api/tags")
        response.raise_for_status()
        return {model["name"]: model["digest"] for model in response.json()["models"]}

    def unload(self, model: str):
        response = self.http.post("/api/generate", json={"model": model, "keep_alive": 0})
        response.raise_for_status()

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        options: dict | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> Generation:
        body = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": options or {},
        }
        if output_format is not None:
            body["format"] = output_format
        started = time.perf_counter()
        response = self.http.post("/api/generate", json=body)
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - started
        if (
            data.get("error")
            or data.get("done") is not True
            or not isinstance(data.get("response"), str)
        ):
            raise ValueError(
                f"Invalid or incomplete Ollama response: {data.get('error', 'missing completion')}"
            )
        return Generation(
            model=model,
            prompt=prompt,
            response=data["response"],
            latency=elapsed,
            load_duration=data.get("load_duration", 0) / 1e9,
            eval_count=data.get("eval_count", 0),
            done_reason=data.get("done_reason", ""),
        )


def generate(model: str, prompt: str) -> Generation:
    """Send one prompt to local Ollama and return response plus wall-clock seconds."""
    client = OllamaClient()
    try:
        return client.generate(model, prompt)
    finally:
        client.close()
