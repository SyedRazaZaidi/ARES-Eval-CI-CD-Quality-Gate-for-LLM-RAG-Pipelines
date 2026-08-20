"""Command-line interface: evaluate, demo, redteam, gate, drift."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ares_eval.config import get_settings, load_thresholds
from ares_eval.evaluators.heuristic_judge import HeuristicJudgeEvaluator
from ares_eval.models.dataset import load_golden_dataset
from ares_eval.models.results import BatchRunSummary
from ares_eval.orchestrator.runner import AresTestRunner
from ares_eval.orchestrator.synthesizer import synthesize_adversarial
from ares_eval.paths import artifacts_dir, config_dir, data_dir
from ares_eval.pipeline.demo_rag import NorthstarRAG
from ares_eval.reporting.exporter import export_run
from ares_eval.telemetry.ledger import EvaluationLedger

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="ARES-Eval — enterprise AI quality & CI/CD regression harness.",
)
console = Console()


def _judge(mode: str):
    settings = get_settings()
    if mode == "offline":
        return HeuristicJudgeEvaluator()
    if mode == "llm":
        from ares_eval.evaluators.llm_judge import LLMJudgeEvaluator

        if not settings.has_llm_credentials():
            raise typer.BadParameter("LLM judge requested but HF_TOKEN / OPENAI_API_KEY is not set.")
        return LLMJudgeEvaluator()
    if settings.has_llm_credentials():
        from ares_eval.evaluators.llm_judge import LLMJudgeEvaluator

        return LLMJudgeEvaluator()
    return HeuristicJudgeEvaluator()


def _target(name: str):
    if name in {"demo", "northstar"}:
        return NorthstarRAG(broken=False)
    if name in {"demo-broken", "broken"}:
        return NorthstarRAG(broken=True)
    if name == "echo":
        return lambda query: query
    if ":" in name:
        module_name, func_name = name.split(":", 1)
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    raise typer.BadParameter(f"Unknown target '{name}'. Use demo, demo-broken, echo, or module:callable.")


def _load_baseline(path: Path | None) -> BatchRunSummary | None:
    if path is None or not path.exists():
        return None
    return BatchRunSummary.model_validate_json(path.read_text(encoding="utf-8"))


def _print_summary(summary: BatchRunSummary) -> None:
    table = Table(title="ARES-Eval gate")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="right")
    status = "[green]PASS[/green]" if summary.is_gate_passed else "[red]FAIL[/red]"
    table.add_row("Gate", status, "")
    table.add_row("Faithfulness", f"{summary.mean_faithfulness:.4f}", "")
    table.add_row("Answer relevance", f"{summary.mean_answer_relevance:.4f}", "")
    table.add_row("Context recall", f"{summary.mean_context_recall:.4f}", "")
    table.add_row("Token F1", f"{summary.mean_token_f1:.4f}", "")
    table.add_row("Hallucination %", f"{summary.hallucination_rate_pct:.2f}", "")
    table.add_row("Injection success %", f"{summary.injection_success_rate_pct:.2f}", "")
    table.add_row("P95 latency ms", f"{summary.p95_latency_ms:.1f}", "")
    table.add_row("Passed cases", f"{summary.passed_cases}/{summary.total_cases}", "")
    console.print(table)
    if summary.gate:
        for check in summary.gate.checks:
            mark = "PASS" if check.passed else "FAIL"
            color = "green" if check.passed else "red"
            console.print(
                f"  [{color}]{mark}[/{color}] {check.name}: {check.actual:.4f} {check.comparator} {check.threshold:.4f}"
                + (f" ({check.detail})" if check.detail else "")
            )


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., exists=True, help="Golden dataset JSON"),
    target: str = typer.Option("demo", help="demo | demo-broken | echo | module:callable"),
    output_dir: Path = typer.Option(None, help="Artifact directory"),
    judge: str = typer.Option("auto", help="auto | offline | llm"),
    context_mode: str = typer.Option("pipeline", help="pipeline | gold"),
    baseline: Path | None = typer.Option(None, help="Prior report.json to compare"),
    synthesize: bool = typer.Option(False, help="Append synthesized injection permutations"),
    sample: float = typer.Option(1.0, min=0.05, max=1.0),
    persist: bool = typer.Option(True, help="Write SQLite ledger"),
    thresholds: Path | None = typer.Option(None, help="Override SLO file"),
) -> None:
    """Run the golden suite against a candidate pipeline and enforce the quality gate."""
    gold = load_golden_dataset(dataset)
    if synthesize:
        gold = synthesize_adversarial(gold)
    runner = AresTestRunner(
        inference_callable=_target(target),
        judge=_judge(judge),
        thresholds=load_thresholds(thresholds) if thresholds else load_thresholds(),
        context_mode=context_mode,
    )
    summary = asyncio.run(runner.run_evaluation_suite(gold, _load_baseline(baseline), sample_rate=sample))
    out = output_dir or artifacts_dir()
    paths = export_run(summary, out, _load_baseline(baseline))
    if persist:
        EvaluationLedger(get_settings().ledger_path()).persist(summary)
    _print_summary(summary)
    console.print(f"Wrote [bold]{paths['html']}[/bold] and {paths['markdown']}")
    if not summary.is_gate_passed:
        raise typer.Exit(code=1)


@app.command()
def demo(
    broken: bool = typer.Option(False, help="Use shuffled retrieval to demonstrate a red gate"),
    output_dir: Path = typer.Option(None),
    judge: str = typer.Option("offline"),
) -> None:
    """End-to-end offline demo: Northstar BM25 RAG + golden suite + HTML report."""
    dataset = data_dir() / "golden" / "enterprise_core.json"
    evaluate(
        dataset=dataset,
        target="demo-broken" if broken else "demo",
        output_dir=output_dir or artifacts_dir(),
        judge=judge,
        context_mode="pipeline",
        baseline=None,
        synthesize=False,
        sample=1.0,
        persist=True,
        thresholds=None,
    )


@app.command()
def redteam(
    output_dir: Path = typer.Option(None),
    judge: str = typer.Option("offline"),
    target: str = typer.Option("demo"),
) -> None:
    """Run the adversarial injection / PII / jailbreak suite."""
    dataset = data_dir() / "golden" / "adversarial_injection.json"
    evaluate(
        dataset=dataset,
        target=target,
        output_dir=output_dir or (artifacts_dir() / "redteam"),
        judge=judge,
        context_mode="pipeline",
        baseline=None,
        synthesize=False,
        sample=1.0,
        persist=True,
        thresholds=config_dir() / "thresholds.safety.json",
    )


@app.command("verify-gate")
def verify_gate(
    report: Path = typer.Option(..., exists=True, help="report.json from a previous run"),
) -> None:
    """Re-check a saved report against current thresholds (CI final step)."""
    summary = BatchRunSummary.model_validate_json(report.read_text(encoding="utf-8"))
    from ares_eval.orchestrator.gates import evaluate_gate

    gate = evaluate_gate(summary, load_thresholds())
    summary.gate = gate
    summary.is_gate_passed = gate.passed
    _print_summary(summary)
    if not gate.passed:
        raise typer.Exit(code=1)


@app.command()
def drift(
    days: int = typer.Option(30, min=1),
) -> None:
    """Inspect the local SQLite ledger for faithfulness drift."""
    report = EvaluationLedger(get_settings().ledger_path()).drift_report(days=days)
    console.print_json(json.dumps(report))
    if report.get("status") == "drift":
        raise typer.Exit(code=1)


@app.command()
def synthesize(
    dataset: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
) -> None:
    """Write a dataset expanded with programmatic injection permutations."""
    gold = synthesize_adversarial(load_golden_dataset(dataset))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(gold.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"Wrote {len(gold.test_cases)} cases to {output}")
