"""Self-contained HTML quality report (opens in any browser, no server)."""

from __future__ import annotations

import html
import json

from ares_eval.models.results import BatchRunSummary


def render_html(summary: BatchRunSummary, baseline: BatchRunSummary | None = None) -> str:
    payload = summary.model_dump(mode="json")
    baseline_payload = baseline.model_dump(mode="json") if baseline else None
    banner = "PASSED" if summary.is_gate_passed else "FAILED"
    tone = "#34d399" if summary.is_gate_passed else "#f87171"
    data_json = json.dumps({"current": payload, "baseline": baseline_payload})
    cases_rows = []
    for row in summary.results:
        cls = "ok" if row.passed else "bad"
        cases_rows.append(
            "<tr class='{cls}'><td><code>{tid}</code></td><td>{diff}</td>"
            "<td>{faith:.2f}</td><td>{rel:.2f}</td><td>{f1:.2f}</td>"
            "<td>{hall}</td><td>{flags}</td></tr>".format(
                cls=cls,
                tid=html.escape(row.test_id),
                diff=html.escape(row.difficulty),
                faith=row.faithfulness,
                rel=row.answer_relevance,
                f1=row.token_f1,
                hall="yes" if row.hallucination_detected else "no",
                flags=html.escape(", ".join(row.safety.flags) or "—"),
            )
        )
    table = "\n".join(cases_rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>ARES-Eval · {banner}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg:#0b1220; --card:#121a2b; --ink:#e5eefc; --muted:#8aa0c2;
      --line:#243049; --good:{tone}; --accent:#2dd4bf;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #123, var(--bg));
      color: var(--ink);
    }}
    header {{
      padding: 32px 40px 12px; border-bottom:1px solid var(--line);
      display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
    }}
    h1 {{ margin:0; font-size:28px; letter-spacing:.04em; }}
    .badge {{
      background: color-mix(in srgb, var(--good) 18%, transparent);
      color: var(--good); border:1px solid var(--good);
      padding:8px 14px; border-radius:999px; font-weight:700;
    }}
    main {{ padding: 24px 40px 60px; max-width: 1200px; margin:0 auto; }}
    .kpis {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
    .card {{
      background: var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 18px;
    }}
    .card h3 {{ margin:0 0 6px; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .card .v {{ font-size:26px; font-weight:700; }}
    .grid {{ display:grid; grid-template-columns: 1.1fr .9fr; gap:16px; margin-top:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
    tr.bad td {{ color:#fca5a5; }}
    code {{ color:#7dd3fc; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    canvas {{ max-height: 280px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} header, main {{ padding-left:18px; padding-right:18px; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <div class="muted">ARES-Eval · enterprise AI quality gate</div>
      <h1>Quality report</h1>
      <p class="muted">{html.escape(summary.dataset_name)} · {html.escape(summary.model_version)} · {html.escape(summary.judge_backend)}</p>
    </div>
    <div class="badge">{banner}</div>
  </header>
  <main>
    <section class="kpis">
      <article class="card"><h3>Faithfulness</h3><div class="v">{summary.mean_faithfulness:.3f}</div></article>
      <article class="card"><h3>Relevance</h3><div class="v">{summary.mean_answer_relevance:.3f}</div></article>
      <article class="card"><h3>Hallucination</h3><div class="v">{summary.hallucination_rate_pct:.2f}%</div></article>
      <article class="card"><h3>P95 latency</h3><div class="v">{summary.p95_latency_ms:.0f} ms</div></article>
      <article class="card"><h3>Pass rate</h3><div class="v">{summary.passed_cases}/{summary.total_cases}</div></article>
      <article class="card"><h3>Est. cost</h3><div class="v">${summary.total_cost_usd:.4f}</div></article>
    </section>
    <section class="grid">
      <article class="card"><canvas id="metrics"></canvas></article>
      <article class="card">
        <h3>Run lineage</h3>
        <p>Commit <code>{html.escape(summary.commit_sha[:12])}</code></p>
        <p>Dataset fingerprint <code>{html.escape(summary.dataset_fingerprint[:16])}</code></p>
        <p>Run ID <code>{html.escape(summary.run_id)}</code></p>
        <p class="muted">Faithfulness ≠ correctness: a pipeline can be faithful to the wrong retrieved chunks. Context precision/recall catch that class of regression.</p>
      </article>
    </section>
    <article class="card" style="margin-top:16px; overflow:auto;">
      <h3>Per-case results</h3>
      <table>
        <thead><tr><th>ID</th><th>Difficulty</th><th>Faith</th><th>Rel</th><th>F1</th><th>Halluc.</th><th>Flags</th></tr></thead>
        <tbody>
          {table}
        </tbody>
      </table>
    </article>
  </main>
  <script>
    const DATA = {data_json};
    const cur = DATA.current;
    const base = DATA.baseline;
    const labels = ["Faithfulness","Relevance","Ctx precision","Ctx recall","Token F1"];
    const current = [cur.mean_faithfulness, cur.mean_answer_relevance, cur.mean_context_precision, cur.mean_context_recall, cur.mean_token_f1];
    const baseline = base ? [base.mean_faithfulness, base.mean_answer_relevance, base.mean_context_precision, base.mean_context_recall, base.mean_token_f1] : null;
    const datasets = [{{
      label: "Current candidate",
      data: current,
      backgroundColor: "rgba(45,212,191,.75)"
    }}];
    if (baseline) {{
      datasets.push({{
        label: "Baseline",
        data: baseline,
        backgroundColor: "rgba(148,163,184,.45)"
      }});
    }}
    new Chart(document.getElementById("metrics"), {{
      type: "bar",
      data: {{ labels, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: "#dbeafe" }} }} }},
        scales: {{
          y: {{ min: 0, max: 1, ticks: {{ color: "#8aa0c2" }}, grid: {{ color: "#243049" }} }},
          x: {{ ticks: {{ color: "#8aa0c2" }}, grid: {{ display:false }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
