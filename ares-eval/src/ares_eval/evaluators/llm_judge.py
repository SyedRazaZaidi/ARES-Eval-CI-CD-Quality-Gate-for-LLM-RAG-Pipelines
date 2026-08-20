"""Schema-validated LLM-as-a-judge over any OpenAI-compatible endpoint."""

from __future__ import annotations

import json

from openai import OpenAI
from pydantic import ValidationError

from ares_eval.config import get_settings
from ares_eval.evaluators.heuristic_judge import HeuristicJudgeEvaluator, JudgeEvaluationOutput
from ares_eval.models.dataset import GoldenTestCase

JUDGE_SYSTEM_PROMPT = """
You are an expert AI Verification Judge in an enterprise software testing harness.
Evaluate the candidate model answer against retrieved contexts and the ground-truth reference.

Strict protocol:
1. Extract every factual assertion from the Model Answer.
2. Mark an assertion as Hallucinated if it is not entailed by Retrieved Contexts.
3. Faithfulness = supported assertions / total assertions (0 to 1).
4. Answer relevance = how well the answer addresses the user query (0 to 1).
5. Out-of-domain questions should score high relevance if the model refuses or says it does not know.
6. Output MUST be valid JSON with keys:
   faithfulness_score, hallucination_detected, hallucinated_claims,
   factual_claims_verified, answer_relevance_score, reasoning_summary.
""".strip()

# Rough public-router pricing placeholder used only for ledger estimates.
_USD_PER_1K_TOKENS = 0.0008


class LLMJudgeEvaluator:
    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.huggingface_api_key or settings.openai_api_key
        if settings.huggingface_api_key:
            base_url = settings.hf_base_url
        else:
            base_url = settings.openai_base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.request_timeout_seconds)
        self.model = settings.judge_model_name
        self.retries = settings.judge_max_retries
        self._fallback = HeuristicJudgeEvaluator()

    def evaluate_sample(
        self,
        query: str,
        contexts: list[str],
        model_answer: str,
        ground_truth: str,
        case: GoldenTestCase | None = None,
    ) -> JudgeEvaluationOutput:
        payload = {
            "user_query": query,
            "retrieved_contexts": contexts,
            "model_answer": model_answer,
            "reference_ground_truth": ground_truth,
        }
        last_error = ""
        for _attempt in range(self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"EVALUATE CANDIDATE RUN:\n{json.dumps(payload)}",
                        },
                    ],
                )
                raw = response.choices[0].message.content or "{}"
                parsed = json.loads(raw)
                result = JudgeEvaluationOutput.model_validate(parsed)
                usage = getattr(response, "usage", None)
                prompt_toks = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_toks = int(getattr(usage, "completion_tokens", 0) or 0)
                result.tokens_prompt = prompt_toks
                result.tokens_completion = completion_toks
                result.cost_usd = round(((prompt_toks + completion_toks) / 1000) * _USD_PER_1K_TOKENS, 6)
                result.backend = "llm"
                return result
            except (json.JSONDecodeError, ValidationError, OSError, Exception) as exc:  # noqa: BLE001
                last_error = str(exc)
                continue
        fallback = self._fallback.evaluate_sample(query, contexts, model_answer, ground_truth, case)
        fallback.reasoning_summary = (
            f"LLM judge failed ({last_error[:180]}). Fell back to heuristic. "
            + fallback.reasoning_summary
        )
        fallback.backend = "hybrid"
        return fallback
