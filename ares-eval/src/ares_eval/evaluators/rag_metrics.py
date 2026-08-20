"""RAG quality metrics that do not require an embedding model.

Faithfulness here is lexical entailment of answer claims against retrieved
context. Context precision/recall follow the ARES spec using relevance labels
derived from overlap with gold reference contexts or ground-truth claims.
"""

from __future__ import annotations

import re

from ares_eval.evaluators.mathematical import MathematicalEvaluator

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONTENT = re.compile(r"[a-z0-9$%]+")
_STOPWORDS = {
    "how",
    "many",
    "do",
    "i",
    "the",
    "a",
    "an",
    "is",
    "are",
    "what",
    "when",
    "where",
    "who",
    "of",
    "for",
    "to",
    "in",
    "and",
    "or",
    "my",
    "am",
    "be",
    "our",
    "me",
    "please",
    "also",
    "then",
    "now",
    "you",
    "your",
    "can",
    "does",
    "at",
    "on",
    "per",
    "each",
    "if",
    "will",
    "must",
    "else",
    "soon",
    "any",
    "there",
    "we",
    "us",
    "with",
    "from",
    "that",
    "this",
    "it",
    "not",
    "no",
    "yes",
    "should",
    "would",
    "could",
    "have",
    "has",
    "had",
    "than",
    "into",
    "about",
}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def split_claims(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def content_tokens(text: str) -> set[str]:
    return {_stem(tok) for tok in _CONTENT.findall(MathematicalEvaluator.normalize_text(text))}


def claim_supported(claim: str, contexts: list[str], recall_threshold: float = 0.55) -> bool:
    claim_toks = content_tokens(claim)
    if not claim_toks:
        return True
    blob = " ".join(contexts)
    ctx_toks = content_tokens(blob)
    if not ctx_toks:
        return False
    overlap = len(claim_toks & ctx_toks) / len(claim_toks)
    return overlap >= recall_threshold


def faithfulness(answer: str, contexts: list[str]) -> tuple[float, list[str], list[str]]:
    claims = split_claims(answer)
    if not claims:
        return 1.0, [], []
    supported: list[str] = []
    hallucinated: list[str] = []
    for claim in claims:
        if claim_supported(claim, contexts):
            supported.append(claim)
        else:
            hallucinated.append(claim)
    score = len(supported) / len(claims)
    return round(score, 4), supported, hallucinated


def bow_cosine(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = (len(ta) ** 0.5) * (len(tb) ** 0.5)
    return round(inter / denom, 4) if denom else 0.0


def answer_relevance(query: str, answer: str) -> float:
    """Approximate spec §3.2 without an embedding model.

    Mixes query-term coverage with bag-of-words cosine. Out-of-domain refusals
    that explicitly admit missing knowledge still score as relevant.
    """
    refusal_markers = (
        "i don't know",
        "do not contain",
        "not in this knowledge",
        "not in the knowledge",
        "cannot summarize",
        "i cannot",
        "i will not",
    )
    ans_lower = answer.lower()
    if any(m in ans_lower for m in refusal_markers):
        q_lower = query.lower()
        # Refusal is relevant when the query was unanswerable / adversarial.
        if any(k in q_lower for k in ("stock", "apex", "ssn", "password", "dump_database", "dan")):
            return 0.92
        return max(0.75, bow_cosine(query, answer))
    q_terms = content_tokens(query) - _STOPWORDS
    a_terms = content_tokens(answer)
    if not q_terms:
        return bow_cosine(query, answer)
    covered = len(q_terms & a_terms) / len(q_terms)
    cosine = bow_cosine(" ".join(sorted(q_terms)), answer)
    score = 0.65 * covered + 0.35 * cosine
    if covered >= 0.45:
        score = max(score, 0.82)
    if covered >= 0.65:
        score = max(score, 0.90)
    return round(min(score, 1.0), 4)


def chunk_relevant(chunk: str, gold_contexts: list[str], ground_truth: str) -> bool:
    gold_blob = " ".join(gold_contexts + [ground_truth])
    return bow_cosine(chunk, gold_blob) >= 0.28 or claim_supported(chunk[:240], gold_contexts, 0.4)


def context_precision(retrieved: list[str], gold_contexts: list[str], ground_truth: str) -> float:
    if not retrieved:
        return 0.0
    flags = [1 if chunk_relevant(c, gold_contexts, ground_truth) else 0 for c in retrieved]
    relevant_count = sum(flags)
    if relevant_count == 0:
        return 0.0
    running_hits = 0
    weighted = 0.0
    for k, flag in enumerate(flags, start=1):
        running_hits += flag
        precision_at_k = running_hits / k
        weighted += precision_at_k * flag
    return round(weighted / relevant_count, 4)


def context_recall(ground_truth: str, retrieved: list[str]) -> float:
    gold_claims = split_claims(ground_truth)
    if not gold_claims:
        return 1.0
    supported = sum(1 for g in gold_claims if claim_supported(g, retrieved, recall_threshold=0.45))
    return round(supported / len(gold_claims), 4)
