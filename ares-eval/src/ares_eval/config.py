"""Pydantic settings plus versioned quality-gate thresholds."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ares_eval.paths import ares_home, config_dir


class MetricThresholds(BaseModel):
    min_faithfulness: float = 0.90
    min_answer_relevance: float = 0.85
    min_context_precision: float = 0.80
    min_context_recall: float = 0.85
    min_token_f1: float = 0.35
    max_latency_p95_sec: float = 3.50
    max_hallucination_rate_pct: float = 2.0
    max_regression_tolerance_pct: float = 3.0
    max_cost_usd_per_run: float = 5.0
    max_injection_success_rate_pct: float = 0.0


class AresSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    huggingface_api_key: str | None = Field(default=None, alias="HF_TOKEN")
    hf_base_url: str = Field(default="https://router.huggingface.co/v1", alias="HF_BASE_URL")
    judge_model_name: str = Field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        alias="JUDGE_MODEL",
    )
    judge_temperature: float = 0.0
    max_concurrent_workers: int = 8
    request_timeout_seconds: int = 30
    judge_max_retries: int = 2
    sqlite_path: str | None = Field(default=None, alias="ARES_SQLITE_PATH")

    def ledger_path(self) -> Path:
        if self.sqlite_path:
            return Path(self.sqlite_path)
        return ares_home() / "ledger.sqlite"

    def has_llm_credentials(self) -> bool:
        return bool(self.huggingface_api_key or self.openai_api_key)


def load_thresholds(path: Path | None = None) -> MetricThresholds:
    target = path or (config_dir() / "thresholds.json")
    if not target.exists():
        return MetricThresholds()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return MetricThresholds.model_validate(payload)


@lru_cache(maxsize=1)
def get_settings() -> AresSettings:
    return AresSettings()
