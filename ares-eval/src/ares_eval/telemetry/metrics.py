"""Prometheus textfile metrics for scrape-without-a-server demos."""

from __future__ import annotations

from pathlib import Path

from ares_eval.models.results import BatchRunSummary


def render_prometheus(summary: BatchRunSummary) -> str:
    labels = (
        f'dataset="{summary.dataset_name}",'
        f'backend="{summary.judge_backend}",'
        f'gate="{"pass" if summary.is_gate_passed else "fail"}"'
    )
    lines = [
        "# HELP ares_eval_faithfulness_mean Mean faithfulness of the latest run.",
        "# TYPE ares_eval_faithfulness_mean gauge",
        f"ares_eval_faithfulness_mean{{{labels}}} {summary.mean_faithfulness}",
        "# HELP ares_eval_hallucination_rate_pct Hallucination rate percent.",
        "# TYPE ares_eval_hallucination_rate_pct gauge",
        f"ares_eval_hallucination_rate_pct{{{labels}}} {summary.hallucination_rate_pct}",
        "# HELP ares_eval_latency_p95_ms End-to-end candidate latency p95.",
        "# TYPE ares_eval_latency_p95_ms gauge",
        f"ares_eval_latency_p95_ms{{{labels}}} {summary.p95_latency_ms}",
        "# HELP ares_eval_gate_passed 1 if the quality gate passed.",
        "# TYPE ares_eval_gate_passed gauge",
        f"ares_eval_gate_passed{{{labels}}} {1 if summary.is_gate_passed else 0}",
        "# HELP ares_eval_cost_usd Estimated judge spend for the run.",
        "# TYPE ares_eval_cost_usd gauge",
        f"ares_eval_cost_usd{{{labels}}} {summary.total_cost_usd}",
        "# HELP ares_eval_cases Number of cases in the run.",
        "# TYPE ares_eval_cases gauge",
        f"ares_eval_cases{{{labels}}} {summary.total_cases}",
    ]
    return "\n".join(lines) + "\n"


def write_prometheus(summary: BatchRunSummary, path: Path) -> None:
    path.write_text(render_prometheus(summary), encoding="utf-8")
