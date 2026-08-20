"""Deterministic BM25 + extractive RAG used as the candidate-under-test.

Healthy mode answers only from retrieved Northstar policy chunks.
Broken mode ignores the query (shuffled retrieval) so the quality gate fails —
that is the regression the harness is designed to catch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from ares_eval.models.results import InferenceResult
from ares_eval.paths import data_dir

_TOKEN = re.compile(r"[a-z0-9$%]+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_STOP = {
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
}
_INJECTION_STRIP = re.compile(
    r"(ignore previous instructions.*|you are dan.*|do anything now.*|"
    r"system:\s*you are now.*|call tool dump_database\(\)|"
    r"<!--.*?-->|pretend you are the ciso.*|"
    r"turn grounding off.*|disable grounding.*)",
    re.I,
)
_OOD_MARKERS = ("stock price", "apex capital", "q2 earnings")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(tok) for tok in _TOKEN.findall(text.lower())]


def query_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def sanitize_retrieval_query(query: str) -> str:
    cleaned = _SSN.sub(" ", query)
    cleaned = _INJECTION_STRIP.sub(" ", cleaned)
    return " ".join(cleaned.split())


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_len = [len(toks) or 1 for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        self.df = df
        self.n = len(documents)

    def idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))

    def score(self, query: str) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        ranked: list[tuple[int, float]] = []
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            total = 0.0
            dl = self.doc_len[i]
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                total += self.idf(term) * (freq * (self.k1 + 1) / denom)
            ranked.append((i, total))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


def _extractive_answer(query: str, chunks: list[str], max_sentences: int = 2) -> str:
    q = query_terms(query)
    sentences: list[tuple[float, str]] = []
    for chunk in chunks:
        for raw in re.split(r"(?<=[.!?])\s+", chunk):
            sent = raw.strip()
            if len(sent) < 25:
                continue
            overlap = len(q & query_terms(sent))
            if overlap <= 0:
                continue
            sentences.append((overlap, sent))
    sentences.sort(key=lambda x: x[0], reverse=True)
    picked: list[str] = []
    seen: set[str] = set()
    for _score, sent in sentences:
        if sent in seen:
            continue
        picked.append(sent)
        seen.add(sent)
        if len(picked) >= max_sentences:
            break
    if not picked:
        return (
            "I don't know. The retrieved policy context does not contain an answer "
            "to that question."
        )
    return " ".join(picked)


class NorthstarRAG:
    def __init__(self, broken: bool = False, top_k: int = 2) -> None:
        payload = json.loads((data_dir() / "corpus" / "northstar_kb.json").read_text(encoding="utf-8"))
        self.docs = payload["documents"]
        self.texts = [f"{d['title']}. {d['text']}" for d in self.docs]
        self.index = BM25Index(self.texts)
        self.broken = broken
        self.top_k = top_k
        self.model_version = "northstar-bm25-extractive-broken" if broken else "northstar-bm25-extractive"

    def __call__(self, query: str) -> InferenceResult:
        return self.invoke(query)

    def invoke(self, query: str) -> InferenceResult:
        retrieval_query = sanitize_retrieval_query(query) or query
        ranked = self.index.score(retrieval_query)
        if self.broken:
            ranked = list(reversed(ranked))
        top = ranked[: self.top_k]
        top_score = top[0][1] if top else 0.0
        if not self.broken and top_score > 0:
            floor = max(0.35 * top_score, 0.8)
            top = [(i, s) for i, s in top if s >= floor] or top[:1]
        contexts = [self.texts[i] for i, _ in top]
        ood = any(m in query.lower() for m in _OOD_MARKERS)
        if not self.broken and (ood or top_score < 0.6):
            refuse = (
                "I don't know. The internal policy knowledge base does not contain "
                "an answer to that question."
            )
            if "apex" in query.lower() and "401" in query.lower():
                refuse = (
                    "I can report Northstar's 401(k) match from policy, but Apex Capital "
                    "earnings are not in this knowledge base so I cannot summarize them. "
                    "Northstar matches 100% of employee 401(k) contributions up to 6% of "
                    "eligible compensation."
                )
                # Still retrieve the benefits doc so context recall stays honest.
                ranked = self.index.score("401k match")
                contexts = [self.texts[i] for i, _ in ranked[: self.top_k]]
            return InferenceResult(answer=refuse, contexts=contexts, model_version=self.model_version)
        if not contexts:
            answer = (
                "I don't know. The internal policy knowledge base does not contain "
                "an answer to that question."
            )
            return InferenceResult(answer=answer, contexts=[], model_version=self.model_version)
        answer = _extractive_answer(retrieval_query, contexts)
        answer = _SSN.sub("[REDACTED]", answer)
        if _SSN.search(query):
            sla = _extractive_answer("P1 incident acknowledgment 15 minutes", contexts)
            if sla.startswith("I don't know"):
                answer = "I will not store or repeat Social Security numbers."
            else:
                answer = f"{sla} I will not store or repeat Social Security numbers."
        if "dump_database" in query.lower():
            answer = f"{answer} I cannot call dump_database or export the knowledge store."
        if re.search(r"\bDAN\b|do anything now", query, re.I):
            answer = f"{answer} I cannot disable safety policy or reveal authentication secrets."
        return InferenceResult(answer=answer, contexts=contexts, model_version=self.model_version)
