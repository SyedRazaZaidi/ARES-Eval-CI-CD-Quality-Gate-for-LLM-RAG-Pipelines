"""SQLite historical ledger — no server process required."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ares_eval.models.results import BatchRunSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    commit_sha TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    judge_backend TEXT NOT NULL,
    total_samples INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    mean_faithfulness REAL NOT NULL,
    mean_answer_relevance REAL NOT NULL,
    mean_token_f1 REAL NOT NULL,
    mean_context_precision REAL NOT NULL,
    mean_context_recall REAL NOT NULL,
    hallucination_rate_pct REAL NOT NULL,
    injection_success_rate_pct REAL NOT NULL,
    p95_latency_ms REAL NOT NULL,
    total_cost_usd REAL NOT NULL,
    gate_passed INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_sample_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    test_id TEXT NOT NULL,
    query TEXT NOT NULL,
    model_output TEXT NOT NULL,
    faithfulness_score REAL,
    hallucination_detected INTEGER,
    passed INTEGER,
    reasoning TEXT,
    tags TEXT,
    created_at TEXT NOT NULL
);
"""


class EvaluationLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def persist(self, summary: BatchRunSummary) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, commit_sha, branch_name, model_version, dataset_name,
                    dataset_fingerprint, judge_backend, total_samples, passed_cases,
                    mean_faithfulness, mean_answer_relevance, mean_token_f1,
                    mean_context_precision, mean_context_recall, hallucination_rate_pct,
                    injection_success_rate_pct, p95_latency_ms, total_cost_usd,
                    gate_passed, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    summary.run_id,
                    summary.commit_sha,
                    summary.branch_name,
                    summary.model_version,
                    summary.dataset_name,
                    summary.dataset_fingerprint,
                    summary.judge_backend,
                    summary.total_cases,
                    summary.passed_cases,
                    summary.mean_faithfulness,
                    summary.mean_answer_relevance,
                    summary.mean_token_f1,
                    summary.mean_context_precision,
                    summary.mean_context_recall,
                    summary.hallucination_rate_pct,
                    summary.injection_success_rate_pct,
                    summary.p95_latency_ms,
                    summary.total_cost_usd,
                    int(summary.is_gate_passed),
                    now,
                ),
            )
            for row in summary.results:
                conn.execute(
                    """
                    INSERT INTO evaluation_sample_records (
                        run_id, test_id, query, model_output, faithfulness_score,
                        hallucination_detected, passed, reasoning, tags, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        summary.run_id,
                        row.test_id,
                        row.query,
                        row.model_answer,
                        row.faithfulness,
                        int(row.hallucination_detected),
                        int(row.passed),
                        row.reasoning,
                        json.dumps(row.tags),
                        now,
                    ),
                )

    def recent_runs(self, days: int = 30) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluation_runs
                WHERE created_at >= ?
                ORDER BY created_at DESC
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def drift_report(self, days: int = 30, faithfulness_floor: float = 0.90) -> dict:
        runs = self.recent_runs(days)
        if len(runs) < 2:
            return {
                "status": "insufficient_history",
                "runs": len(runs),
                "message": "Need at least two stored runs to compute drift.",
            }
        newest, oldest = runs[0], runs[-1]
        delta = newest["mean_faithfulness"] - oldest["mean_faithfulness"]
        below = [r for r in runs if r["mean_faithfulness"] < faithfulness_floor]
        return {
            "status": "drift" if delta < -0.03 or below else "stable",
            "runs": len(runs),
            "newest_faithfulness": newest["mean_faithfulness"],
            "oldest_faithfulness": oldest["mean_faithfulness"],
            "delta": round(delta, 4),
            "runs_below_slo": len(below),
        }
