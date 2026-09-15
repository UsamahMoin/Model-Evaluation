from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL REFERENCES runs(run_id), model TEXT NOT NULL, test_id TEXT NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY (run_id, model, test_id)
);
"""


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)

    def start(self, run_id: str, created_at: str, metadata: dict):
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs VALUES (?, ?, 'running', ?)",
                (run_id, created_at, json.dumps(metadata)),
            )

    def save(self, row: dict):
        with self.connection:
            self.connection.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?)",
                (row["run_id"], row["model"], row["test_id"], json.dumps(row)),
            )

    def finish(self, run_id: str, status: str, metadata: dict | None = None):
        with self.connection:
            self.connection.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
            if metadata is not None:
                self.connection.execute(
                    "UPDATE runs SET metadata = ? WHERE run_id = ?", (json.dumps(metadata), run_id)
                )

    def frame(self, run_id: str) -> pd.DataFrame:
        rows = self.connection.execute(
            "SELECT payload FROM results WHERE run_id = ? ORDER BY model, test_id", (run_id,)
        ).fetchall()
        return pd.DataFrame([json.loads(row[0]) for row in rows])

    def close(self):
        self.connection.close()


def write_csv(frame: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_results(path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, dtype={"test_id": str, "model": str})
    required = {
        "run_id",
        "model",
        "test_id",
        "category",
        "passed",
        "score",
        "latency",
        "status",
        "failure_type",
        "scoring_hash",
        "evaluator_version",
        "suite",
        "run_complete",
        "expected_count",
    }
    missing = required - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Invalid result file: empty or missing columns {sorted(missing)}")
    for column in (
        "passed",
        "run_complete",
        *(["dataset_complete"] if "dataset_complete" in frame else []),
    ):
        values = frame[column].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"{column} must contain true/false values")
        frame[column] = values == "true"
    for column in ("score", "latency", "expected_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].map(math.isfinite).all():
            raise ValueError(f"{column} must contain finite values")
    if not frame.score.between(0, 1).all() or not frame.latency.ge(0).all():
        raise ValueError("Scores must be 0..1 and latencies nonnegative")
    if not frame.status.isin(["ok", "generation_error", "evaluation_error"]).all():
        raise ValueError("Unknown result status")
    if ((frame.status != "ok") & frame.passed).any():
        raise ValueError("An errored result cannot pass")
    if frame.duplicated(["model", "test_id"]).any() or frame.run_id.nunique() != 1:
        raise ValueError("Each file must contain one run with unique model/test pairs")
    return frame
