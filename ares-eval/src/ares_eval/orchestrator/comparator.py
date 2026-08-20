"""Bootstrap confidence intervals and baseline comparison."""

from __future__ import annotations

import random
from collections.abc import Sequence

from ares_eval.models.results import BatchRunSummary, ConfidenceInterval


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 800,
    alpha: float = 0.05,
    seed: int = 7,
) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(mean=0.0, low=0.0, high=0.0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return ConfidenceInterval(mean=round(mean, 4), low=round(mean, 4), high=round(mean, 4))
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo_idx = int(alpha / 2 * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return ConfidenceInterval(
        mean=round(mean, 4),
        low=round(samples[lo_idx], 4),
        high=round(samples[hi_idx], 4),
    )


def metric_deltas(current: BatchRunSummary, baseline: BatchRunSummary) -> dict[str, float]:
    return {
        "faithfulness": round(current.mean_faithfulness - baseline.mean_faithfulness, 4),
        "answer_relevance": round(current.mean_answer_relevance - baseline.mean_answer_relevance, 4),
        "token_f1": round(current.mean_token_f1 - baseline.mean_token_f1, 4),
        "hallucination_rate_pct": round(
            current.hallucination_rate_pct - baseline.hallucination_rate_pct, 2
        ),
        "p95_latency_ms": round(current.p95_latency_ms - baseline.p95_latency_ms, 2),
    }
