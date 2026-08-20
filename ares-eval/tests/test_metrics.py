from __future__ import annotations

from ares_eval.evaluators.mathematical import MathematicalEvaluator
from ares_eval.evaluators.rag_metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)


def test_exact_match_normalization() -> None:
    ev = MathematicalEvaluator
    assert ev.compute_exact_match("The Cat.", "the cat") == 1.0
    assert ev.compute_exact_match("dog", "cat") == 0.0


def test_token_f1_known_value() -> None:
    f1, precision, recall = MathematicalEvaluator.compute_token_f1("a red cat", "the red cat")
    assert precision > 0.6
    assert recall > 0.6
    assert 0.7 < f1 <= 1.0


def test_bleu_and_rouge_identical() -> None:
    text = "Northstar matches six percent of compensation."
    ev = MathematicalEvaluator
    assert ev.compute_bleu(text, text) == 1.0
    assert ev.compute_rouge_l(text, text) == 1.0


def test_faithfulness_detects_unsupported_claim() -> None:
    score, _ok, hallucinated = faithfulness(
        "Employees receive 20 days of PTO. The CEO lives on Mars.",
        ["Northstar provides 20 days of paid time off per calendar year."],
    )
    assert score < 1.0
    assert hallucinated


def test_context_precision_ranks_relevant_first() -> None:
    gold = ["20 days of PTO per calendar year"]
    retrieved = [
        "Northstar provides 20 days of PTO per calendar year to full-time employees.",
        "The cafeteria serves tacos on Tuesdays.",
    ]
    score = context_precision(retrieved, gold, "20 days of PTO")
    assert score > 0.7


def test_context_recall_requires_coverage() -> None:
    gt = "P1 incidents must be acknowledged within 15 minutes."
    assert context_recall(gt, ["P1 incidents require acknowledgment within 15 minutes."]) == 1.0
    assert context_recall(gt, ["The office has three floors."]) < 0.5


def test_ood_refusal_is_relevant() -> None:
    score = answer_relevance(
        "What will Northstar's stock price be next quarter?",
        "I don't know. The internal policy knowledge base does not contain stock price forecasts.",
    )
    assert score >= 0.75
