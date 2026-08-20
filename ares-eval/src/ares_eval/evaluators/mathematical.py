"""Deterministic lexical metrics: exact match, token F1, BLEU, ROUGE-L."""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


class MathMetricResult(BaseModel):
    exact_match: float
    token_f1: float
    token_precision: float
    token_recall: float
    bleu: float
    rouge_l: float


class MathematicalEvaluator:
    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = _ARTICLES.sub(" ", text)
        text = _PUNCT.sub(" ", text)
        return " ".join(text.split())

    @classmethod
    def tokens(cls, text: str) -> list[str]:
        return cls.normalize_text(text).split()

    @classmethod
    def compute_exact_match(cls, prediction: str, ground_truth: str) -> float:
        return 1.0 if cls.normalize_text(prediction) == cls.normalize_text(ground_truth) else 0.0

    @classmethod
    def compute_token_f1(cls, prediction: str, ground_truth: str) -> tuple[float, float, float]:
        pred_tokens = cls.tokens(prediction)
        gt_tokens = cls.tokens(ground_truth)
        if not pred_tokens or not gt_tokens:
            return 0.0, 0.0, 0.0
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0, 0.0, 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1, 4), round(precision, 4), round(recall, 4)

    @classmethod
    def compute_bleu(cls, prediction: str, ground_truth: str, max_n: int = 4) -> float:
        pred = cls.tokens(prediction)
        ref = cls.tokens(ground_truth)
        if not pred or not ref:
            return 0.0
        precisions: list[float] = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(_ngrams(pred, n))
            ref_ngrams = Counter(_ngrams(ref, n))
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = max(sum(pred_ngrams.values()), 1)
            precisions.append(overlap / total)
        if min(precisions) == 0:
            geo = 0.0
        else:
            geo = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        bp = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(len(pred), 1))
        return round(bp * geo, 4)

    @classmethod
    def compute_rouge_l(cls, prediction: str, ground_truth: str) -> float:
        pred = cls.tokens(prediction)
        ref = cls.tokens(ground_truth)
        if not pred or not ref:
            return 0.0
        lcs = _lcs_length(pred, ref)
        precision = lcs / len(pred)
        recall = lcs / len(ref)
        if precision + recall == 0:
            return 0.0
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1, 4)

    @classmethod
    def entity_recall(cls, prediction: str, expected_entities: list[str]) -> float:
        if not expected_entities:
            return 1.0
        haystack = cls.normalize_text(prediction)
        hits = sum(1 for ent in expected_entities if cls.normalize_text(ent) in haystack)
        return round(hits / len(expected_entities), 4)

    @classmethod
    def evaluate(cls, prediction: str, ground_truth: str) -> MathMetricResult:
        f1, precision, recall = cls.compute_token_f1(prediction, ground_truth)
        return MathMetricResult(
            exact_match=cls.compute_exact_match(prediction, ground_truth),
            token_f1=f1,
            token_precision=precision,
            token_recall=recall,
            bleu=cls.compute_bleu(prediction, ground_truth),
            rouge_l=cls.compute_rouge_l(prediction, ground_truth),
        )


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]
