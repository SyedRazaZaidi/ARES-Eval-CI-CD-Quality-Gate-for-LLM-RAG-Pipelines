"""Persist JSON / markdown / HTML / Prometheus artifacts."""

from __future__ import annotations

from pathlib import Path

from ares_eval.models.results import BatchRunSummary
from ares_eval.reporting.html_gen import render_html
from ares_eval.reporting.markdown_gen import render_markdown
from ares_eval.telemetry.metrics import write_prometheus


def export_run(
    summary: BatchRunSummary,
    output_dir: Path,
    baseline: BatchRunSummary | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "report.json",
        "markdown": output_dir / "pr_comment.md",
        "html": output_dir / "report.html",
        "prometheus": output_dir / "metrics.prom",
    }
    paths["json"].write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(summary, baseline), encoding="utf-8")
    paths["html"].write_text(render_html(summary, baseline), encoding="utf-8")
    write_prometheus(summary, paths["prometheus"])
    return paths
