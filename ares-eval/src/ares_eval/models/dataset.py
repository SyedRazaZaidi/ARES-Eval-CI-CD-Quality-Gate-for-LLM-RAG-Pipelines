"""Typed golden-dataset models with JSON Schema validation on load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from jsonschema import Draft7Validator
from pydantic import BaseModel, Field

from ares_eval.paths import data_dir

Difficulty = Literal["simple", "multi-hop", "adversarial", "out-of-domain"]
ExpectedBehavior = Literal["answer", "refuse", "answer_original", "no_pii"]


class GoldenTestCase(BaseModel):
    id: str
    query: str = Field(min_length=5)
    reference_contexts: list[str] = Field(min_length=1)
    ground_truth_answer: str = Field(min_length=5)
    expected_entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(min_length=1)
    difficulty: Difficulty = "simple"
    expected_behavior: ExpectedBehavior = "answer"
    injection_payload: str | None = None
    parent_id: str | None = None


class GoldenDataset(BaseModel):
    version: str
    domain: str
    description: str = ""
    test_cases: list[GoldenTestCase] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        from ares_eval.telemetry.fingerprint import sha256_canonical

        return sha256_canonical(self.model_dump(mode="json"))


def load_schema() -> dict:
    schema_path = data_dir() / "schemas" / "golden_schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    target = Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    validator = Draft7Validator(load_schema())
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        where = ".".join(str(p) for p in first.path) or "<root>"
        raise ValueError(f"Golden dataset schema error at {where}: {first.message}")
    return GoldenDataset.model_validate(raw)
