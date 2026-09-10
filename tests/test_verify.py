"""
Tests for heinzy/generation/verify.py -- the Treatment-arm second-pass
verifier. Fully offline: a fake generator stands in for Generator, no
Ollama/Azure/network call is ever made.
"""
from __future__ import annotations

import inspect
import json

import pytest

from heinzy.generation.generator import Answer
from heinzy.generation.verify import (
    VERIFIER_PROMPT_VERSION,
    VerificationResult,
    verifier_prompt_fingerprint,
    verify_answer,
)
from heinzy.retrieval.store import ScoredChunk

# The v1 fingerprint recorded in the 30-question pilot report
# (eval/results/pilot_full_30q_20260908T174032Z.json). v2's prompt text must
# differ from v1's, so the fingerprint must differ from this specific known
# historical value -- not just "be some 12-char string".
V1_FINGERPRINT = "a3f39f0348ce"

HITS = [
    ScoredChunk(
        chunk_id="c1",
        text="Students typically take 54 units per semester.",
        score=0.81,
        doc_id="doc-x",
        section_path="8.1. Normal Courseload Expectation",
        source_pages=[11],
    ),
]

DRAFT = Answer(
    query="How many units per semester?",
    text="Students typically take 54 units per semester.",
    model_tag="fake-gen-model",
    sources=HITS,
    refused=False,
    cited_sections=[],
    unsupported_citations=[],
)

REFUSED_LAYER1_DRAFT = Answer(
    query="How many units per semester?",
    text="I can't answer that from the MISM handbook.",
    model_tag="fake-gen-model",
    sources=[],
    refused=True,
    refusal_reason="no_retrieved_context",
    raw_text=None,
)


class _FakeGenerator:
    """Stands in for heinzy.generation.generator.Generator's chat plumbing."""

    def __init__(self, content: str, provider: str = "ollama",
                 model_tag: str = "fake-gen-model",
                 refusal_text: str = "STANDARD PROJECT REFUSAL TEXT",
                 raw_extra: dict | None = None) -> None:
        self.provider = provider
        self.model_tag = model_tag
        self.refusal_text = refusal_text
        self._content = content
        self._raw_extra = raw_extra or {}
        self.calls: list[dict] = []

    def _chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        data = {"message": {"content": self._content}}
        data.update(self._raw_extra)
        return data


def _json(decision: str, final_text: str = "x", reason: str = "because") -> str:
    return json.dumps({"decision": decision, "final_text": final_text, "reason": reason})


# --------------------------------------------------------------------------- #
# 1-3: valid decisions parse
# --------------------------------------------------------------------------- #
def test_valid_keep_json_parses():
    gen = _FakeGenerator(_json("KEEP", final_text="ignored", reason="supported"))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "KEEP"
    assert result.parse_error is None


def test_valid_revise_json_parses():
    gen = _FakeGenerator(_json("REVISE", final_text="corrected answer", reason="draft missed a fact"))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "REVISE"
    assert result.parse_error is None


def test_valid_abstain_json_parses():
    gen = _FakeGenerator(_json("ABSTAIN", reason="insufficient evidence"))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "ABSTAIN"
    assert result.parse_error is None


# --------------------------------------------------------------------------- #
# 4-6: decision -> final_text/refused semantics
# --------------------------------------------------------------------------- #
def test_keep_returns_original_draft_text_exactly():
    gen = _FakeGenerator(_json("KEEP", final_text="a paraphrase the verifier made up"))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.final_text == DRAFT.text
    assert result.final_text != "a paraphrase the verifier made up"


def test_revise_uses_verifier_final_text_with_no_second_generation_call():
    gen = _FakeGenerator(_json("REVISE", final_text='Actually 54 units (see "8.1").'))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "REVISE"
    assert result.final_text == 'Actually 54 units (see "8.1").'
    assert result.refused is False
    assert len(gen.calls) == 1, "verifier must make exactly one call, no separate draft call"


def test_abstain_sets_refused_true_and_uses_standard_text():
    gen = _FakeGenerator(_json("ABSTAIN", final_text="the verifier's own prose"),
                          refusal_text="STANDARD PROJECT REFUSAL TEXT")
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.refused is True
    assert result.final_text == "STANDARD PROJECT REFUSAL TEXT"
    assert result.final_text != "the verifier's own prose"


def test_abstain_makes_no_extra_model_call():
    """Same one-call guarantee REVISE already has (see
    test_revise_uses_verifier_final_text_with_no_second_generation_call),
    checked explicitly for ABSTAIN too: no third model call, ever."""
    gen = _FakeGenerator(_json("ABSTAIN"))
    verify_answer("q", DRAFT, HITS, gen)
    assert len(gen.calls) == 1


# --------------------------------------------------------------------------- #
# 7: refused draft still verified
# --------------------------------------------------------------------------- #
def test_refused_draft_is_still_sent_through_verification():
    gen = _FakeGenerator(_json("REVISE", final_text='It is 54 units (see "8.1").'))
    result = verify_answer("q", REFUSED_LAYER1_DRAFT, HITS, gen)
    assert len(gen.calls) == 1
    assert result.decision == "REVISE"
    assert result.refused is False
    # the draft's refusal status must appear in the prompt sent to the model
    prompt = gen.calls[0]["messages"][1]["content"]
    assert "declined to answer" in prompt


def test_keep_on_a_refused_draft_preserves_refused_true():
    gen = _FakeGenerator(_json("KEEP", reason="context genuinely insufficient"))
    result = verify_answer("q", REFUSED_LAYER1_DRAFT, HITS, gen)
    assert result.decision == "KEEP"
    assert result.refused is True
    assert result.final_text == REFUSED_LAYER1_DRAFT.text


# --------------------------------------------------------------------------- #
# 8-9: malformed output fails safely
# --------------------------------------------------------------------------- #
def test_malformed_json_falls_back_to_abstain_safely():
    gen = _FakeGenerator("this is not json at all", refusal_text="STANDARD TEXT")
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "ABSTAIN"
    assert result.refused is True
    assert result.final_text == "STANDARD TEXT"
    assert result.parse_error is not None
    assert result.raw_verifier_output == "this is not json at all"


def test_missing_decision_falls_back_to_abstain_safely():
    gen = _FakeGenerator(json.dumps({"final_text": "x", "reason": "y"}), refusal_text="STANDARD TEXT")
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "ABSTAIN"
    assert result.refused is True
    assert result.final_text == "STANDARD TEXT"
    assert result.parse_error is not None


def test_invalid_decision_value_falls_back_to_abstain_safely():
    gen = _FakeGenerator(_json("MAYBE"), refusal_text="STANDARD TEXT")
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "ABSTAIN"
    assert result.parse_error is not None


def test_revise_missing_final_text_falls_back_to_abstain_safely():
    gen = _FakeGenerator(json.dumps({"decision": "REVISE", "reason": "y"}), refusal_text="STANDARD TEXT")
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "ABSTAIN"
    assert result.final_text == "STANDARD TEXT"
    assert result.parse_error is not None


def test_json_wrapped_in_prose_still_parses():
    """Models sometimes ignore 'JSON only' and add a sentence around it."""
    wrapped = f"Sure, here is my answer:\n{_json('KEEP')}\nHope that helps."
    gen = _FakeGenerator(wrapped)
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "KEEP"


@pytest.mark.parametrize(
    "content",
    [
        _json("KEEP"),
        _json("REVISE", final_text="corrected"),
        _json("ABSTAIN"),
        "not json at all",  # malformed -> fallback ABSTAIN
        json.dumps({"reason": "y"}),  # missing decision -> fallback ABSTAIN
    ],
)
def test_model_call_made_is_true_on_every_outcome(content):
    """The _chat call always happens before any parsing, on every return
    path including the conservative fallback -- so model_call_made is never
    False for a VerificationResult, unlike a Layer 1 Answer.usage."""
    gen = _FakeGenerator(content)
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.model_call_made is True


# --------------------------------------------------------------------------- #
# 10: no Row/gold fields can reach the verifier
# --------------------------------------------------------------------------- #
def test_verifier_signature_accepts_no_row_or_gold_fields():
    params = list(inspect.signature(verify_answer).parameters)
    assert params == ["question", "draft", "hits", "generator"]
    forbidden = (
        "row", "answerable", "gold_answer", "gold_claims", "unanswerable_reason",
        "question_type", "supporting_evidence", "policy_unit_id", "question_family_id",
    )
    for name in forbidden:
        assert name not in params


def test_verifier_never_calls_generate():
    """verify_answer must use only _chat, never generate() (which would
    produce a second, forbidden draft). The fake deliberately has no
    generate method, so an accidental call here would raise AttributeError."""
    gen = _FakeGenerator(_json("KEEP"))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.decision == "KEEP"


def test_verifier_calls_chat_with_no_tools():
    gen = _FakeGenerator(_json("KEEP"))
    verify_answer("q", DRAFT, HITS, gen)
    assert gen.calls[0]["tools"] is None


# --------------------------------------------------------------------------- #
# 11-12: usage metadata
# --------------------------------------------------------------------------- #
def test_usage_captured_when_provider_reports_it():
    gen = _FakeGenerator(_json("KEEP"), raw_extra={"prompt_eval_count": 100, "eval_count": 25})
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.total_tokens == 125


def test_usage_is_none_when_provider_does_not_report_it():
    gen = _FakeGenerator(_json("KEEP"))  # no raw_extra
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.total_tokens is None


def test_azure_shaped_usage_is_captured():
    gen = _FakeGenerator(
        _json("KEEP"), provider="azure_openai",
        raw_extra={"raw": {"usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}}},
    )
    result = verify_answer("q", DRAFT, HITS, gen)
    assert result.input_tokens == 50
    assert result.output_tokens == 10
    assert result.total_tokens == 60


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def test_latency_is_recorded_as_nonnegative_float():
    gen = _FakeGenerator(_json("KEEP"))
    result = verify_answer("q", DRAFT, HITS, gen)
    assert isinstance(result.latency_seconds, float)
    assert result.latency_seconds >= 0


def test_prompt_fingerprint_is_stable_and_short():
    a = verifier_prompt_fingerprint()
    b = verifier_prompt_fingerprint()
    assert a == b
    assert len(a) == 12


def test_prompt_version_is_v2():
    assert VERIFIER_PROMPT_VERSION == "v2"


def test_prompt_fingerprint_changed_from_v1():
    assert verifier_prompt_fingerprint() != V1_FINGERPRINT


def test_verification_result_is_immutable():
    result = VerificationResult(
        decision="KEEP", final_text="x", refused=False, reason="y",
        raw_verifier_output="z", model_tag="m", input_tokens=None,
        output_tokens=None, total_tokens=None, latency_seconds=0.0,
        model_call_made=True,
    )
    with pytest.raises(Exception):
        result.decision = "REVISE"
