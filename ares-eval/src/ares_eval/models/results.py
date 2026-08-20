"""Evaluation telemetry containers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JudgeBackend = Literal["heuristic", "llm", "hybrid"]


class InferenceResult(BaseModel):
    """Normalized output from a candidate pipeline under test."""

    answer: str
    contexts: list[str] = Field(default_factory=list)
    model_version: str = "unknown"
    extra: dict[str, Any] = Field(default_factory=dict)


class SafetyVerdict(BaseModel):
    injection_attempt: bool = False
    injection_succeeded: bool = False
    pii_leaked: bool = False
    jailbreak_language_followed: bool = False
    flags: list[str] = Field(default_factory=list)


class TestCaseResult(BaseModel):
    test_id: str
    query: str
    model_answer: str
    ground_truth: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    latency_ms: float
    exact_match: float
    token_f1: float
    token_precision: float = 0.0
    token_recall: float = 0.0
    bleu: float = 0.0
    rouge_l: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float
    answer_relevance: float
    hallucination_detected: bool
    safety: SafetyVerdict = Field(default_factory=SafetyVerdict)
    judge_backend: JudgeBackend = "heuristic"
    reasoning: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "simple"
    expected_behavior: str = "answer"
    passed: bool = False
    entity_recall: float = 0.0


class SliceScore(BaseModel):
    name: str
    count: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_token_f1: float
    pass_rate: float
    hallucination_rate_pct: float


class ConfidenceInterval(BaseModel):
    mean: float
    low: float
    high: float


class GateCheck(BaseModel):
    name: str
    passed: bool
    actual: float
    threshold: float
    comparator: str
    detail: str = ""


class GateDecision(BaseModel):
    passed: bool
    checks: list[GateCheck]
    regression_failed: bool = False


class BatchRunSummary(BaseModel):
    run_id: str
    commit_sha: str = "unknown"
    branch_name: str = "unknown"
    model_version: str = "unknown"
    dataset_name: str
    dataset_fingerprint: str
    judge_backend: JudgeBackend
    total_cases: int
    passed_cases: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_token_f1: float
    mean_bleu: float = 0.0
    mean_rouge_l: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    hallucination_rate_pct: float
    injection_success_rate_pct: float = 0.0
    p95_latency_ms: float
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    faithfulness_ci: ConfidenceInterval | None = None
    is_gate_passed: bool
    gate: GateDecision | None = None
    slices: list[SliceScore] = Field(default_factory=list)
    results: list[TestCaseResult]
