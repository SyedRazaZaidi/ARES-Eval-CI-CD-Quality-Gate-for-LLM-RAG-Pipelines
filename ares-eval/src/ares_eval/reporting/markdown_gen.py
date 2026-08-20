"""GitHub PR markdown report."""

from __future__ import annotations

from ares_eval.models.results import BatchRunSummary, GateCheck


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _status(check: GateCheck | None, fallback_ok: bool) -> str:
    if check is None:
        return "✅ Passed" if fallback_ok else "❌ Failed"
    return "✅ Passed" if check.passed else "❌ Failed"


def _find(summary: BatchRunSummary, name: str) -> GateCheck | None:
    if not summary.gate:
        return None
    for check in summary.gate.checks:
        if check.name == name:
            return check
    return None


def render_markdown(summary: BatchRunSummary, baseline: BatchRunSummary | None = None) -> str:
    banner = "PASSED ✅" if summary.is_gate_passed else "FAILED ❌"
    lines = [
        f"## 📊 ARES-Eval Quality Gate: {banner}",
        "",
        f"Commit: `{summary.commit_sha[:12]}` · Dataset: `{summary.dataset_name}` "
        f"· Judge: `{summary.judge_backend}` · Model: `{summary.model_version}`",
        f"Fingerprint: `{summary.dataset_fingerprint[:16]}` · Run: `{summary.run_id}`",
        "",
        "| Quality dimension | Current | Baseline | Delta | Gate |",
        "|---|---:|---:|---:|---|",
    ]

    def row(label: str, key: str, current: float, fmt: str, higher_better: bool = True) -> None:
        prior = None
        if baseline is not None:
            prior = getattr(baseline, key)
        delta = "" if prior is None else (current - prior)
        if fmt == "pct":
            cur_s = _pct(current)
            base_s = "—" if prior is None else _pct(prior)
            delta_s = "—" if prior is None else f"{delta * 100:+.2f}pp"
        elif fmt == "pp":
            cur_s = f"{current:.2f}%"
            base_s = "—" if prior is None else f"{prior:.2f}%"
            delta_s = "—" if prior is None else f"{delta:+.2f}pp"
        else:
            cur_s = f"{current:.2f}"
            base_s = "—" if prior is None else f"{prior:.2f}"
            delta_s = "—" if prior is None else f"{delta:+.2f}"
        check = _find(summary, key if key != "hallucination_rate_pct" else "hallucination_rate_pct")
        ok = True if check is None else check.passed
        lines.append(f"| {label} | {cur_s} | {base_s} | {delta_s} | {_status(check, ok)} |")

    row("Faithfulness (groundedness)", "mean_faithfulness", summary.mean_faithfulness, "pct")
    row("Answer relevance", "mean_answer_relevance", summary.mean_answer_relevance, "pct")
    row("Context precision", "mean_context_precision", summary.mean_context_precision, "pct")
    row("Context recall", "mean_context_recall", summary.mean_context_recall, "pct")
    row("Token F1", "mean_token_f1", summary.mean_token_f1, "pct")
    row("Hallucination rate", "hallucination_rate_pct", summary.hallucination_rate_pct, "pp", False)
    row("Injection success", "injection_success_rate_pct", summary.injection_success_rate_pct, "pp", False)
    row("Latency P95 (ms)", "p95_latency_ms", summary.p95_latency_ms, "num", False)

    ci = summary.faithfulness_ci
    if ci:
        lines += [
            "",
            f"Faithfulness 95% bootstrap CI: **{ci.low:.3f} – {ci.high:.3f}** (mean {ci.mean:.3f}).",
        ]
    lines += [
        "",
        f"Cases: **{summary.passed_cases}/{summary.total_cases} passed** · "
        f"Est. judge cost: **${summary.total_cost_usd:.4f}** · Tokens: **{summary.total_tokens}**",
        "",
    ]
    if summary.slices:
        lines += ["### Slice scores", "", "| Slice | N | Faithfulness | Pass rate |", "|---|---:|---:|---:|"]
        for sl in summary.slices:
            if sl.name.startswith("difficulty:") or sl.name.startswith("tag:safety"):
                lines.append(
                    f"| `{sl.name}` | {sl.count} | {_pct(sl.mean_faithfulness)} | {_pct(sl.pass_rate)} |"
                )
        lines.append("")

    failures = [r for r in summary.results if not r.passed]
    if failures:
        lines += ["### Failures", ""]
        for row_res in failures[:12]:
            reason = row_res.reasoning.replace("\n", " ")[:180]
            flags = ", ".join(row_res.safety.flags) or "quality"
            lines.append(f"- `{row_res.test_id}` ({flags}): {reason}")
        lines.append("")
    else:
        lines.append("Zero regression anomalies detected. Ready for merge.")
    return "\n".join(lines) + "\n"
