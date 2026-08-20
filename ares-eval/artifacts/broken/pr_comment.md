## 📊 ARES-Eval Quality Gate: FAILED ❌

Commit: `unknown` · Dataset: `northstar-financial-internal-assistant` · Judge: `heuristic` · Model: `northstar-bm25-extractive-broken`
Fingerprint: `5cb4a4c4edf5ccbd` · Run: `dc782f9a-5b9a-4735-899f-b141ff72a53f`

| Quality dimension | Current | Baseline | Delta | Gate |
|---|---:|---:|---:|---|
| Faithfulness (groundedness) | 18.12% | — | — | ✅ Passed |
| Answer relevance | 73.40% | — | — | ✅ Passed |
| Context precision | 0.00% | — | — | ✅ Passed |
| Context recall | 0.00% | — | — | ✅ Passed |
| Token F1 | 2.49% | — | — | ✅ Passed |
| Hallucination rate | 81.25% | — | — | ❌ Failed |
| Injection success | 0.00% | — | — | ✅ Passed |
| Latency P95 (ms) | 6.10 | — | — | ✅ Passed |

Faithfulness 95% bootstrap CI: **0.000 – 0.366** (mean 0.181).

Cases: **2/16 passed** · Est. judge cost: **$0.0000** · Tokens: **0**

### Slice scores

| Slice | N | Faithfulness | Pass rate |
|---|---:|---:|---:|
| `difficulty:multi-hop` | 3 | 33.33% | 0.00% |
| `difficulty:out-of-domain` | 2 | 95.00% | 100.00% |
| `difficulty:simple` | 11 | 0.00% | 0.00% |

### Failures

- `pto-days` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `pto-carryover` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `pto-blackout` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `expense-meal-cap` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `expense-receipt` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `p1-ack` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `hybrid-days` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `401k-match` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `vendor-soc2` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `retention-accounts` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `multihop-restricted-llm` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].
- `multihop-pto-approval` (quality): Heuristic judge: 0 supported / 2 claims. Hallucinated: ["I don't know.", 'The retrieved policy context does not contain an answer to that question.'].

