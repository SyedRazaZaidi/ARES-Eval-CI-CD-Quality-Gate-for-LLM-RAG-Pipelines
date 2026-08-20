## 📊 ARES-Eval Quality Gate: PASSED ✅

Commit: `unknown` · Dataset: `northstar-financial-internal-assistant` · Judge: `heuristic` · Model: `northstar-bm25-extractive`
Fingerprint: `5cb4a4c4edf5ccbd` · Run: `54a8460d-7a34-42f6-8037-289a1a68c33d`

| Quality dimension | Current | Baseline | Delta | Gate |
|---|---:|---:|---:|---|
| Faithfulness (groundedness) | 99.37% | — | — | ✅ Passed |
| Answer relevance | 89.00% | — | — | ✅ Passed |
| Context precision | 100.00% | — | — | ✅ Passed |
| Context recall | 89.29% | — | — | ✅ Passed |
| Token F1 | 65.29% | — | — | ✅ Passed |
| Hallucination rate | 0.00% | — | — | ✅ Passed |
| Injection success | 0.00% | — | — | ✅ Passed |
| Latency P95 (ms) | 9.02 | — | — | ✅ Passed |

Faithfulness 95% bootstrap CI: **0.984 – 1.000** (mean 0.994).

Cases: **16/16 passed** · Est. judge cost: **$0.0000** · Tokens: **0**

### Slice scores

| Slice | N | Faithfulness | Pass rate |
|---|---:|---:|---:|
| `difficulty:multi-hop` | 3 | 100.00% | 100.00% |
| `difficulty:out-of-domain` | 2 | 95.00% | 100.00% |
| `difficulty:simple` | 11 | 100.00% | 100.00% |

Zero regression anomalies detected. Ready for merge.
