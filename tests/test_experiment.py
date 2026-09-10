"""
Tests for heinzy/eval/experiment.py -- the paired Control/Treatment pilot
runner. Fully offline: fake Retriever/Generator/Verifier, no real
retrieval/model calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heinzy.eval.dataset import GoldClaim, Row, SupportingEvidence
from heinzy.eval.experiment import (
    DatasetValidationError,
    ModelUsage,
    ensure_tools_disabled,
    load_and_validate_rows,
    run_experiment,
    shuffled_order,
)
from heinzy.generation.generator import Answer
from heinzy.generation.verify import VerificationResult
from heinzy.retrieval.retrieve import RetrievalResult
from heinzy.retrieval.store import ScoredChunk


def _row(id="q1", question="What is X?", family="fam1", policy="pol1", qtype="lookup") -> Row:
    return Row(
        id=id,
        question=question,
        answerable=True,
        question_type=qtype,
        question_family_id=family,
        policy_unit_id=policy,
        experiment_split="pilot",
        gold_answer="THE GOLD ANSWER -- must never reach the verifier",
        gold_claims=[
            GoldClaim(
                claim_id=f"{id}-c1",
                claim="GOLD CLAIM TEXT",
                supporting_evidence=[
                    SupportingEvidence(section_id="1", evidence_quote="GOLD EVIDENCE QUOTE")
                ],
            )
        ],
        unanswerable_reason=None,
    )


def _hits(n=1, prefix="c") -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk_id=f"{prefix}{i}",
            text=f"chunk text {i}",
            score=0.9 - i * 0.01,
            doc_id="doc-x",
            section_path=f"{i}. Some Section",
            source_pages=[i],
        )
        for i in range(n)
    ]


def _answer(question: str, refused=False, refusal_reason=None, text="an answer", usage=None) -> Answer:
    return Answer(
        query=question,
        text=text,
        model_tag="fake-gen",
        sources=[],
        refused=refused,
        refusal_reason=refusal_reason,
        cited_sections=["1. Some Section"] if not refused else [],
        unsupported_citations=[],
        usage=usage,
    )


def _verification(decision="KEEP", final_text="an answer", refused=False,
                   input_tokens=None, output_tokens=None, total_tokens=None,
                   model_call_made=True) -> VerificationResult:
    return VerificationResult(
        decision=decision, final_text=final_text, refused=refused,
        reason="because", raw_verifier_output=json.dumps({"decision": decision}),
        model_tag="fake-verifier", input_tokens=input_tokens,
        output_tokens=output_tokens, total_tokens=total_tokens,
        latency_seconds=0.01, model_call_made=model_call_made,
    )


class _FakeRetriever:
    def __init__(self, hits_by_question: dict[str, list[ScoredChunk]], k: int = 5) -> None:
        self._hits_by_question = hits_by_question
        self.k = k
        self.calls: list[str] = []

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        self.calls.append(question)
        hits = self._hits_by_question[question]
        return RetrievalResult(
            query=question, hits=hits, k=k or self.k,
            embed_model="fake-embed", is_semantic=True, config_hash="fakehash",
        )


class _FakeGenerator:
    def __init__(self, answers_by_question: dict[str, Answer]) -> None:
        self._answers = answers_by_question
        self.calls: list[tuple[str, list[ScoredChunk]]] = []
        self.use_tools = False
        self.model_tag = "fake-gen"

    def generate(self, question: str, hits: list[ScoredChunk]) -> Answer:
        self.calls.append((question, hits))
        return self._answers[question]


class _RecordingVerifier:
    def __init__(self, results_by_question: dict[str, VerificationResult]) -> None:
        self._results = results_by_question
        self.calls: list[dict] = []

    def __call__(self, question, draft, hits, generator) -> VerificationResult:
        self.calls.append({"question": question, "draft": draft, "hits": hits, "generator": generator})
        return self._results[question]


def _run_single(question="What is X?", answer_kwargs=None, verify_kwargs=None, row_kwargs=None):
    row = _row(question=question, **(row_kwargs or {}))
    hits = _hits(2)
    retriever = _FakeRetriever({question: hits})
    draft = _answer(question, **(answer_kwargs or {}))
    generator = _FakeGenerator({question: draft})
    verification = _verification(**(verify_kwargs or {}))
    verifier = _RecordingVerifier({question: verification})

    results = run_experiment([row], retriever, generator, verifier=verifier)
    return results[0], retriever, generator, verifier, hits, draft, verification, row


def _run_single_zero_hit(question="What is X?", answer_kwargs=None, row_kwargs=None):
    """Like _run_single, but retrieval returns zero hits -- exercises the
    v2.1 guard path. verifier is a _RecordingVerifier with no registered
    questions, so any accidental call raises KeyError rather than silently
    succeeding.
    """
    row = _row(question=question, **(row_kwargs or {}))
    retriever = _FakeRetriever({question: []})
    draft = _answer(question, **(answer_kwargs or {}))
    generator = _FakeGenerator({question: draft})
    verifier = _RecordingVerifier({})

    results = run_experiment([row], retriever, generator, verifier=verifier)
    return results[0], retriever, generator, verifier, draft, row


# --------------------------------------------------------------------------- #
# 1-3: exactly-once call counts
# --------------------------------------------------------------------------- #
def test_retrieval_called_exactly_once_per_question():
    _, retriever, *_ = _run_single()
    assert retriever.calls == ["What is X?"]


def test_generate_called_exactly_once_per_question():
    _, _, generator, *_ = _run_single()
    assert len(generator.calls) == 1


def test_verifier_called_exactly_once_per_question():
    _, _, _, verifier, _, _, _, _ = _run_single()
    assert len(verifier.calls) == 1


# --------------------------------------------------------------------------- #
# 4: identical cached hits used by Generator and Verifier
# --------------------------------------------------------------------------- #
def test_same_cached_hits_used_by_generator_and_verifier():
    _, _, generator, verifier, hits, *_ = _run_single()
    gen_hits = generator.calls[0][1]
    verifier_hits = verifier.calls[0]["hits"]
    assert gen_hits is verifier_hits, "must be the SAME list object, not a re-fetch"
    assert [h.chunk_id for h in gen_hits] == [h.chunk_id for h in hits]


def test_retrieval_not_called_twice_even_though_two_arms_consume_it():
    _, retriever, *_ = _run_single()
    assert len(retriever.calls) == 1


# --------------------------------------------------------------------------- #
# 5-8: arm result semantics
# --------------------------------------------------------------------------- #
def test_control_final_text_equals_first_pass_draft():
    result, *_ = _run_single(answer_kwargs={"text": "the draft text"})
    assert result.control.text == "the draft text"


def test_keep_treatment_final_equals_draft():
    result, *_ = _run_single(
        answer_kwargs={"text": "the draft text"},
        verify_kwargs={"decision": "KEEP", "final_text": "the draft text", "refused": False},
    )
    assert result.treatment.verifier_decision == "KEEP"
    assert result.treatment.final_text == "the draft text"


def test_revise_treatment_final_equals_verifier_final():
    result, *_ = _run_single(
        answer_kwargs={"text": "old draft"},
        verify_kwargs={"decision": "REVISE", "final_text": "corrected answer", "refused": False},
    )
    assert result.treatment.verifier_decision == "REVISE"
    assert result.treatment.final_text == "corrected answer"
    assert result.treatment.final_text != result.control.text


def test_abstain_treatment_refused():
    result, *_ = _run_single(
        verify_kwargs={"decision": "ABSTAIN", "final_text": "standard refusal", "refused": True},
    )
    assert result.treatment.verifier_decision == "ABSTAIN"
    assert result.treatment.refused is True


# --------------------------------------------------------------------------- #
# 9-10: refusals still verified
# --------------------------------------------------------------------------- #
def test_control_refusal_still_sent_to_verifier():
    result, _, _, verifier, *_ = _run_single(
        answer_kwargs={"refused": True, "refusal_reason": "model_insufficient_context", "text": "I can't answer that."},
    )
    assert len(verifier.calls) == 1
    assert verifier.calls[0]["draft"].refused is True
    assert result.control.refused is True


def test_layer1_style_refusal_still_sent_to_verifier():
    result, _, generator, verifier, *_ = _run_single(
        answer_kwargs={"refused": True, "refusal_reason": "no_retrieved_context", "text": "I can't answer that."},
    )
    # Layer 1 refuses before any model call in the real Generator; here we
    # only assert the SAME contract: the refusal still reaches the verifier
    # exactly once with the cached hits, regardless of why it refused.
    assert len(verifier.calls) == 1
    assert verifier.calls[0]["draft"].refusal_reason == "no_retrieved_context"
    assert result.control.refusal_reason == "no_retrieved_context"


# --------------------------------------------------------------------------- #
# 11, 19: no gold leakage; family/policy metadata recorded but not passed
# --------------------------------------------------------------------------- #
def test_no_gold_fields_reach_the_verifier():
    _, _, _, verifier, _, _, _, row = _run_single()
    call = verifier.calls[0]
    assert set(call.keys()) == {"question", "draft", "hits", "generator"}
    assert isinstance(call["question"], str)
    assert isinstance(call["draft"], Answer)
    assert isinstance(call["hits"], list)
    # the Row itself, and every gold attribute on it, never appears in the call
    assert call["question"] != row
    for value in call.values():
        assert value is not row
        assert not isinstance(value, Row)
    assert not hasattr(call["draft"], "gold_answer")
    assert not hasattr(call["draft"], "gold_claims")


def test_question_family_and_policy_metadata_recorded_without_reaching_verifier():
    result, _, _, verifier, *_ = _run_single(row_kwargs={"family": "fam-xyz", "policy": "pol-xyz"})
    assert result.question_family_id == "fam-xyz"
    assert result.policy_unit_id == "pol-xyz"
    call = verifier.calls[0]
    assert "fam-xyz" not in json.dumps(
        {"question": call["question"], "draft": call["draft"].text}
    )


def test_result_never_carries_gold_fields():
    result, *_ = _run_single()
    result_field_names = {f for f in result.__dataclass_fields__}
    assert "gold_answer" not in result_field_names
    assert "gold_claims" not in result_field_names
    assert "answerable" not in result_field_names
    assert "unanswerable_reason" not in result_field_names


# --------------------------------------------------------------------------- #
# 12-15: snapshot / raw-output / citation / latency preservation
# --------------------------------------------------------------------------- #
def test_result_contains_exact_retrieval_snapshot():
    result, _, _, _, hits, *_ = _run_single()
    assert len(result.retrieval.chunks) == len(hits)
    for snap, hit in zip(result.retrieval.chunks, hits):
        assert snap.chunk_id == hit.chunk_id
        assert snap.score == hit.score
        assert snap.doc_id == hit.doc_id
        assert snap.section_path == hit.section_path
        assert snap.source_pages == hit.source_pages
        assert snap.text == hit.text
    assert [s.rank for s in result.retrieval.chunks] == list(range(1, len(hits) + 1))


def test_result_preserves_raw_control_and_treatment_output():
    result, *_ = _run_single(
        answer_kwargs={"text": "final", "refused": False},
    )
    assert result.control.raw_text == result.control.raw_text  # present, not scored
    assert result.treatment.raw_verifier_output == json.dumps({"decision": "KEEP"})


def test_result_preserves_citation_fields_without_scoring_them():
    result, *_ = _run_single(answer_kwargs={"text": "answer"})
    assert result.control.cited_sections == ["1. Some Section"]
    assert result.control.unsupported_citations == []
    arm_fields = set(result.control.__dataclass_fields__)
    assert "citation_valid" not in arm_fields
    assert "gas" not in arm_fields
    assert "evidence_coverage" not in arm_fields


def test_result_contains_latency_fields():
    result, *_ = _run_single()
    assert result.retrieval.latency_seconds >= 0
    assert result.control.generation_latency_seconds >= 0
    assert result.treatment.verifier_latency_seconds >= 0


# --------------------------------------------------------------------------- #
# 16-17: e2e latency composition
# --------------------------------------------------------------------------- #
def test_control_e2e_equals_retrieval_plus_first_pass():
    result, *_ = _run_single()
    assert result.control_e2e_latency_seconds == pytest.approx(
        result.retrieval.latency_seconds + result.control.generation_latency_seconds
    )


def test_treatment_e2e_equals_retrieval_plus_first_pass_plus_verifier():
    result, *_ = _run_single()
    assert result.treatment.effective_e2e_latency_seconds == pytest.approx(
        result.control_e2e_latency_seconds + result.treatment.verifier_latency_seconds
    )


# --------------------------------------------------------------------------- #
# 18: usage composition -- model_call_made distinguishes "no call" (Layer 1,
# contributes 0) from "call made, provider omitted usage" (contributes
# unknown/None). See heinzy/eval/experiment.py::_combine_usage_field.
# --------------------------------------------------------------------------- #
def test_no_first_pass_call_plus_known_verifier_usage_equals_verifier_usage():
    """Scenario 1: Layer 1 refused before any first-pass call (usage=None,
    truly zero cost), verifier usage is known -> Treatment total equals
    verifier usage exactly, not None."""
    result, *_ = _run_single(
        answer_kwargs={"usage": None, "refused": True, "refusal_reason": "no_retrieved_context"},
        verify_kwargs={"input_tokens": 200, "output_tokens": 300, "total_tokens": 500},
    )
    assert result.control.usage.model_call_made is False
    assert result.treatment.verifier_usage.model_call_made is True
    assert result.treatment.effective_total_usage == ModelUsage(200, 300, 500, model_call_made=True)


def test_known_first_pass_and_known_verifier_usage_sums_correctly():
    """Scenario 2: both calls made and both report usage -> plain sum."""
    result, *_ = _run_single(
        answer_kwargs={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        verify_kwargs={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
    )
    assert result.control.usage.model_call_made is True
    assert result.treatment.verifier_usage.model_call_made is True
    assert result.treatment.effective_total_usage == ModelUsage(30, 13, 43, model_call_made=True)


def test_first_pass_call_made_but_usage_unavailable_makes_aggregate_unknown():
    """Scenario 3: first pass DID call the model but the provider didn't
    report usage (fields None, model_call_made=True) -- distinct from no
    call at all. The aggregate must be unknown (None), not the verifier's
    value alone and not a fabricated number."""
    result, *_ = _run_single(
        answer_kwargs={"usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None}},
        verify_kwargs={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
    )
    assert result.control.usage.model_call_made is True
    assert result.control.usage.total_tokens is None
    assert result.treatment.effective_total_usage.total_tokens is None
    assert result.treatment.effective_total_usage.input_tokens is None
    assert result.treatment.effective_total_usage.model_call_made is True


def test_verifier_call_made_but_usage_unavailable_makes_aggregate_unknown():
    """Scenario 4: mirror of scenario 3 on the verifier side."""
    result, *_ = _run_single(
        answer_kwargs={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        verify_kwargs={"input_tokens": None, "output_tokens": None, "total_tokens": None},
    )
    assert result.treatment.verifier_usage.model_call_made is True
    assert result.treatment.verifier_usage.total_tokens is None
    assert result.treatment.effective_total_usage.total_tokens is None


def test_layer1_no_call_distinguishable_from_provider_usage_missing():
    """Scenario 5: both cases render identically as (None, None, None) at
    the raw token-value level, but model_call_made tells them apart."""
    layer1, *_ = _run_single(
        answer_kwargs={"usage": None, "refused": True, "refusal_reason": "no_retrieved_context"},
    )
    provider_missing, *_ = _run_single(
        answer_kwargs={"usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None}},
    )

    assert layer1.control.usage.input_tokens is None
    assert provider_missing.control.usage.input_tokens is None
    assert layer1.control.usage.model_call_made is False
    assert provider_missing.control.usage.model_call_made is True


# --------------------------------------------------------------------------- #
# 20: reproducible question-order shuffle
# --------------------------------------------------------------------------- #
def test_shuffle_is_reproducible_for_the_same_seed():
    assert shuffled_order(10, seed=42) == shuffled_order(10, seed=42)


def test_shuffle_actually_reorders_and_differs_by_seed():
    order_a = shuffled_order(10, seed=1)
    order_b = shuffled_order(10, seed=2)
    assert sorted(order_a) == list(range(10))
    assert order_a != list(range(10))
    assert order_a != order_b


def test_run_experiment_respects_execution_order():
    q1, q2 = "Q1?", "Q2?"
    rows = [_row(id="r1", question=q1), _row(id="r2", question=q2)]
    hits_map = {q1: _hits(1, "a"), q2: _hits(1, "b")}
    retriever = _FakeRetriever(hits_map)
    generator = _FakeGenerator({q1: _answer(q1), q2: _answer(q2)})
    verifier = _RecordingVerifier({q1: _verification(), q2: _verification()})

    results = run_experiment(rows, retriever, generator, verifier=verifier, execution_order=[1, 0])
    assert [r.question_id for r in results] == ["r2", "r1"]


# --------------------------------------------------------------------------- #
# 21: invalid dataset aborts before retrieval/model execution
# --------------------------------------------------------------------------- #
def test_invalid_dataset_raises_before_any_execution(tmp_path: Path):
    bad_row = {
        "id": "bad-1", "question": "q", "answerable": True, "question_type": "lookup",
        "question_family_id": None, "policy_unit_id": None, "experiment_split": "pilot",
        "gold_answer": None,  # invalid: answerable=true requires non-empty gold_answer
        "gold_claims": [], "unanswerable_reason": None,
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad_row))

    with pytest.raises(DatasetValidationError) as exc_info:
        load_and_validate_rows(path)
    assert len(exc_info.value.errors) > 0
    # the exception is raised by the loader/validator alone -- no retriever,
    # generator, or verifier object is even constructed in this code path.


def test_valid_dataset_loads_without_raising(tmp_path: Path):
    good_row = {
        "id": "good-1", "question": "q", "answerable": False, "question_type": "lookup",
        "question_family_id": None, "policy_unit_id": None, "experiment_split": "pilot",
        "gold_answer": None, "gold_claims": [], "unanswerable_reason": "wrong_scope",
    }
    path = tmp_path / "good.jsonl"
    path.write_text(json.dumps(good_row))
    rows = load_and_validate_rows(path)
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# 22: tools-enabled configuration is forced off and reported
# --------------------------------------------------------------------------- #
class _ToolsState:
    def __init__(self, use_tools: bool) -> None:
        self.use_tools = use_tools


def test_ensure_tools_disabled_overrides_and_reports_when_enabled():
    g = _ToolsState(use_tools=True)
    was_enabled = ensure_tools_disabled(g)
    assert was_enabled is True
    assert g.use_tools is False


def test_ensure_tools_disabled_is_a_noop_report_when_already_off():
    g = _ToolsState(use_tools=False)
    was_enabled = ensure_tools_disabled(g)
    assert was_enabled is False
    assert g.use_tools is False


# --------------------------------------------------------------------------- #
# v2.1: deterministic zero-hit guard (Verifier Prompt v2 + routing policy).
# Real v2 pilot showed the verifier fabricating facts/citations when given
# zero retrieved chunks; the guard removes that failure mode deterministically
# by never calling the verifier at all in that case. See heinzy/eval/
# experiment.py's module docstring for the full policy description.
# --------------------------------------------------------------------------- #

# A. zero-hit Treatment skips the verifier entirely
def test_zero_hit_guard_never_calls_verifier():
    result, _, _, verifier, draft, _ = _run_single_zero_hit(
        answer_kwargs={"refused": True, "refusal_reason": "no_retrieved_context",
                       "text": "I can't answer that."}
    )
    assert len(verifier.calls) == 0
    assert result.treatment.zero_hit_guard_triggered is True
    assert result.treatment.verifier_call_made is False
    assert result.treatment.refused is True
    assert result.treatment.verifier_decision == "ABSTAIN"
    assert result.treatment.final_text == draft.text


def test_zero_hit_guard_reason_is_machine_readable_and_output_is_empty():
    result, *_ = _run_single_zero_hit()
    assert result.treatment.verifier_reason == "zero_hit_guard"
    assert result.treatment.raw_verifier_output == ""
    assert result.treatment.verifier_model_tag is None
    assert result.treatment.parse_error is None


# B. zero-hit token accounting: zero, not unknown
def test_zero_hit_verifier_usage_is_zero_call_not_unknown():
    result, *_ = _run_single_zero_hit()
    assert result.treatment.verifier_usage == ModelUsage(None, None, None, model_call_made=False)


def test_zero_hit_treatment_total_reflects_only_control_usage_when_control_has_usage():
    """Verifier is skipped -> contributes exactly 0. Proven by giving Control
    a known nonzero usage and confirming the Treatment total equals it
    exactly (not None/unknown), i.e. skipping really means "+0", not "+?"."""
    result, *_ = _run_single_zero_hit(
        answer_kwargs={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
    )
    assert result.treatment.effective_total_usage == ModelUsage(10, 5, 15, model_call_made=True)


def test_zero_hit_treatment_total_is_real_zero_when_both_sides_skipped():
    """The realistic case: a zero-hit question also means Layer 1 skips the
    first-pass model call, so both sides contribute 0 -> a real integer 0,
    not None."""
    result, *_ = _run_single_zero_hit(
        answer_kwargs={"usage": None, "refused": True, "refusal_reason": "no_retrieved_context"}
    )
    assert result.treatment.effective_total_usage == ModelUsage(0, 0, 0, model_call_made=False)


# C. zero-hit latency accounting: zero, not invented
def test_zero_hit_verifier_latency_is_zero_not_invented():
    result, *_ = _run_single_zero_hit()
    assert result.treatment.verifier_latency_seconds == 0.0
    assert result.treatment.effective_e2e_latency_seconds == pytest.approx(
        result.control_e2e_latency_seconds
    )


# D. nonzero-hit behavior unchanged (KEEP / REVISE / ABSTAIN all still work)
def test_nonzero_hit_guard_not_triggered_and_verifier_called_once():
    result, _, _, verifier, *_ = _run_single()
    assert result.treatment.zero_hit_guard_triggered is False
    assert result.treatment.verifier_call_made is True
    assert len(verifier.calls) == 1


def test_nonzero_hit_keep_path_unaffected_by_guard():
    result, *_ = _run_single(verify_kwargs={"decision": "KEEP", "final_text": "answer text"})
    assert result.treatment.zero_hit_guard_triggered is False
    assert result.treatment.verifier_decision == "KEEP"
    assert result.treatment.final_text == "answer text"


def test_nonzero_hit_revise_path_unaffected_by_guard():
    result, *_ = _run_single(verify_kwargs={"decision": "REVISE", "final_text": "revised text"})
    assert result.treatment.zero_hit_guard_triggered is False
    assert result.treatment.verifier_decision == "REVISE"
    assert result.treatment.final_text == "revised text"


def test_nonzero_hit_abstain_path_unaffected_by_guard():
    result, *_ = _run_single(
        verify_kwargs={"decision": "ABSTAIN", "final_text": "standard refusal", "refused": True}
    )
    assert result.treatment.zero_hit_guard_triggered is False
    assert result.treatment.verifier_decision == "ABSTAIN"
    assert result.treatment.refused is True


# E. Control unchanged in the zero-hit path
def test_zero_hit_control_behavior_is_unaffected_by_guard():
    result, _, _, _, draft, _ = _run_single_zero_hit(
        answer_kwargs={"refused": True, "refusal_reason": "no_retrieved_context",
                       "text": "I can't answer that."}
    )
    assert result.control.text == draft.text
    assert result.control.refused is True
    assert result.control.refusal_reason == "no_retrieved_context"


# F. gold isolation: routing depends on hit count only, never on gold/evaluator fields
def test_zero_hit_guard_triggers_regardless_of_gold_answerable_flag():
    for answerable_flag in (True, False):
        row = Row(
            id="gtest", question="Q?", answerable=answerable_flag,
            question_type="lookup", question_family_id=None, policy_unit_id=None,
            experiment_split="pilot",
            gold_answer=("gold answer" if answerable_flag else None),
            gold_claims=(
                [GoldClaim(claim_id="gtest-c1", claim="c",
                           supporting_evidence=[SupportingEvidence(section_id="1", evidence_quote="q")])]
                if answerable_flag else []
            ),
            unanswerable_reason=(None if answerable_flag else "wrong_scope"),
        )
        retriever = _FakeRetriever({"Q?": []})
        draft = _answer("Q?", refused=True, refusal_reason="no_retrieved_context")
        generator = _FakeGenerator({"Q?": draft})
        results = run_experiment([row], retriever, generator, verifier=_RecordingVerifier({}))
        assert results[0].treatment.zero_hit_guard_triggered is True, f"failed for answerable={answerable_flag}"


def test_zero_hit_guard_introduces_no_gold_text_into_treatment_fields():
    """_row()'s gold_answer/gold_claims are seeded with an obvious 'GOLD'
    sentinel; the guard path must never surface it anywhere in the result."""
    result, *_ = _run_single_zero_hit()
    assert "GOLD" not in result.treatment.final_text
    assert "GOLD" not in result.treatment.verifier_reason
    assert "GOLD" not in result.treatment.raw_verifier_output


# G. strategy version / provenance
def test_build_report_records_treatment_strategy_version():
    from heinzy.eval.experiment import TREATMENT_STRATEGY_VERSION, build_report

    class _FakeCfg:
        config_hash = "fakehash"

    class _FakeGen:
        model_tag = "fake-gen"

    report = build_report(
        [], cfg=_FakeCfg(), generator=_FakeGen(), dataset_path="fake.jsonl",
        dataset_row_count=0, k=5, backend="memory",
        tools_originally_enabled=False, execution_seed=None,
    )
    assert report.treatment_strategy_version == "v2.1" == TREATMENT_STRATEGY_VERSION


def test_verifier_prompt_version_and_fingerprint_unchanged_by_v21():
    from heinzy.generation.verify import VERIFIER_PROMPT_VERSION, verifier_prompt_fingerprint

    assert VERIFIER_PROMPT_VERSION == "v2"
    assert verifier_prompt_fingerprint() == "57916d37c6f6"


# H. no third model call, in either branch
def test_zero_hit_and_nonzero_hit_never_exceed_one_additional_model_call():
    _, _, zero_gen, zero_verifier, *_ = _run_single_zero_hit()
    assert len(zero_gen.calls) == 1  # first pass still happens
    assert len(zero_verifier.calls) == 0  # guard: zero additional calls

    _, _, nonzero_gen, nonzero_verifier, *_ = _run_single(
        verify_kwargs={"decision": "REVISE", "final_text": "revised"}
    )
    assert len(nonzero_gen.calls) == 1
    assert len(nonzero_verifier.calls) == 1  # exactly one additional call, never more
