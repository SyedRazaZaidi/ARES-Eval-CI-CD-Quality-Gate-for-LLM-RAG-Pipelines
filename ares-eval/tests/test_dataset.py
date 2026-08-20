from __future__ import annotations

from ares_eval.evaluators.safety import evaluate_safety
from ares_eval.models.dataset import load_golden_dataset
from ares_eval.orchestrator.synthesizer import synthesize_adversarial
from ares_eval.paths import data_dir


def test_golden_schema_loads() -> None:
    core = load_golden_dataset(data_dir() / "golden" / "enterprise_core.json")
    red = load_golden_dataset(data_dir() / "golden" / "adversarial_injection.json")
    assert core.version == "1.0.0"
    assert len(core.test_cases) >= 12
    assert all(case.tags for case in red.test_cases)
    assert core.fingerprint
    assert core.fingerprint != red.fingerprint


def test_synthesizer_appends_injection_cases() -> None:
    core = load_golden_dataset(data_dir() / "golden" / "enterprise_core.json")
    expanded = synthesize_adversarial(core)
    assert len(expanded.test_cases) > len(core.test_cases)
    synth = [c for c in expanded.test_cases if c.parent_id]
    assert synth
    assert all(c.difficulty == "adversarial" for c in synth)


def test_safety_flags_ssn_echo() -> None:
    red = load_golden_dataset(data_dir() / "golden" / "adversarial_injection.json")
    case = next(c for c in red.test_cases if c.id == "inj-exfil-ssn")
    bad = evaluate_safety(case, "Sure, your SSN is 078-05-1120 and I stored it.")
    assert bad.pii_leaked
    good = evaluate_safety(case, "P1 incidents must be acknowledged within 15 minutes. I will not store SSNs.")
    assert not good.pii_leaked
