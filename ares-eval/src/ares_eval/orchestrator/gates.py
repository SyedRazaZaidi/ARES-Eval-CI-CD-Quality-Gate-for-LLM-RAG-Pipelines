"""Quality-gate policy: SLOs plus optional baseline regression tolerance."""

from __future__ import annotations

from ares_eval.config import MetricThresholds
from ares_eval.models.results import BatchRunSummary, GateCheck, GateDecision


def evaluate_gate(
    summary: BatchRunSummary,
    thresholds: MetricThresholds,
    baseline: BatchRunSummary | None = None,
) -> GateDecision:
    checks: list[GateCheck] = []

    def ge(name: str, actual: float, threshold: float) -> None:
        checks.append(
            GateCheck(
                name=name,
                passed=actual + 1e-12 >= threshold,
                actual=actual,
                threshold=threshold,
                comparator=">=",
            )
        )

    def le(name: str, actual: float, threshold: float) -> None:
        checks.append(
            GateCheck(
                name=name,
                passed=actual <= threshold + 1e-12,
                actual=actual,
                threshold=threshold,
                comparator="<=",
            )
        )

    ge("faithfulness", summary.mean_faithfulness, thresholds.min_faithfulness)
    ge("answer_relevance", summary.mean_answer_relevance, thresholds.min_answer_relevance)
    ge("context_precision", summary.mean_context_precision, thresholds.min_context_precision)
    ge("context_recall", summary.mean_context_recall, thresholds.min_context_recall)
    ge("token_f1", summary.mean_token_f1, thresholds.min_token_f1)
    le("hallucination_rate_pct", summary.hallucination_rate_pct, thresholds.max_hallucination_rate_pct)
    le("latency_p95_sec", summary.p95_latency_ms / 1000.0, thresholds.max_latency_p95_sec)
    le("cost_usd", summary.total_cost_usd, thresholds.max_cost_usd_per_run)
    le(
        "injection_success_rate_pct",
        summary.injection_success_rate_pct,
        thresholds.max_injection_success_rate_pct,
    )

    regression_failed = False
    if baseline is not None:
        pairs = [
            ("faithfulness", summary.mean_faithfulness, baseline.mean_faithfulness, False),
            ("answer_relevance", summary.mean_answer_relevance, baseline.mean_answer_relevance, False),
            ("token_f1", summary.mean_token_f1, baseline.mean_token_f1, False),
            ("hallucination_rate_pct", summary.hallucination_rate_pct, baseline.hallucination_rate_pct, True),
        ]
        tol = thresholds.max_regression_tolerance_pct / 100.0
        for name, current, prior, lower_is_better in pairs:
            if lower_is_better:
                drop = current - prior  # increase is bad
                failed = drop > thresholds.max_regression_tolerance_pct
                detail = f"delta={drop:+.2f}pp vs baseline {prior:.2f}"
            else:
                drop = prior - current  # decrease is bad
                failed = drop > tol
                detail = f"delta={current - prior:+.4f} vs baseline {prior:.4f}"
            if failed:
                regression_failed = True
            checks.append(
                GateCheck(
                    name=f"regression:{name}",
                    passed=not failed,
                    actual=current,
                    threshold=prior,
                    comparator="no-regression",
                    detail=detail,
                )
            )

    passed = all(c.passed for c in checks)
    return GateDecision(passed=passed, checks=checks, regression_failed=regression_failed)
