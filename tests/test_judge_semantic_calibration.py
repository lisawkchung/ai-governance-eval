"""
Regression tests for the v1.2 semantic calibration patch (Judge prompt/rubric
clarifications for Completeness numeric-omission, necessary entailment,
Correctness evidence policy, Faithfulness/Citation-Support deterministic
derivation, and numeric-quantity identity).

IMPORTANT SCOPE NOTE: Task Completion/Correctness/Completeness/Faithfulness/
Citation Support are semantic axes graded BY THE LLM JUDGE, not by any
deterministic function in this codebase -- there is no local "completeness
checker" to unit test in the usual sense, and this task makes zero real
Judge/LLM calls. So each scenario below is verified at the two levels that
actually ARE deterministic and offline-testable:

  1. POLICY TEXT: the v1.2 prompt/rubric documents describe the intended
     rule (content assertions against the actual shipped files).
  2. PIPELINE: build_evaluation_subject/render_judge_request expose exactly
     the premises a judge would need to apply that rule correctly (e.g. for
     the q010-Treatment-style case, BOTH the total-units premise AND the
     per-semester-cap premise actually appear in the rendered input); and
     evaluate_subject(), given a SCRIPTED (never real) judge response that
     already reflects the expected calibration outcome, correctly parses,
     schema-validates, and records that outcome -- proving the runner/schema
     can represent every expected label combination this calibration
     introduces, without silently rejecting or corrupting any of them.

No test in this file makes a network/HTTP call or depends on
eval/annotation/final_human_labels_v2.1.csv (that file is never opened here
or anywhere in heinzy/eval/judge.py -- see that module's own docstring).
"""
from __future__ import annotations

import json

import pytest

from heinzy.eval.dataset import GoldClaim, Row, SupportingEvidence, load_dataset
from heinzy.eval.experiment import RetrievalChunkSnapshot
from heinzy.eval.judge import (
    EXECUTION_SUCCESS,
    ARM_CONTROL,
    EvaluationSubject,
    FlattenedResponse,
    ModelConfig,
    build_evaluation_subject,
    build_gold_index,
    compute_design_provenance,
    evaluate_subject,
    load_judge_prompt_variants,
    load_output_schema,
    render_judge_request,
    validate_judge_output,
)

PROMPT_V12_PATH = "eval/judge/judge_prompt_v1.2.txt"
RUBRIC_V12_PATH = "eval/judge/judge_rubric_v1.2.md"
SCHEMA_PATH = "eval/judge/judge_output_schema_v1.json"
GOLD_V13_PATH = "eval/questions_pilot_v1.3.jsonl"


class FakeJudgeClient:
    """Scripted fake -- never a real model call. Mirrors tests/test_judge.py's
    fixture so this file stays independently runnable."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def run(self, system_prompt, user_prompt, *, temperature, think_value=None,
            response_schema=None, timeout=None):
        from heinzy.eval.judge import JudgeCallResult

        self.calls.append((system_prompt, user_prompt))
        item = self._responses.pop(0)
        return JudgeCallResult(
            raw_text=item, input_tokens=10, output_tokens=5, total_tokens=15,
            model_call_made=True, latency_seconds=0.001, model_tag="fake-judge",
            thinking_trace_present=False,
        )


@pytest.fixture(scope="module")
def prompt_variants_v12():
    return load_judge_prompt_variants(PROMPT_V12_PATH)


@pytest.fixture(scope="module")
def schema_doc():
    return load_output_schema(SCHEMA_PATH)


def _model_config():
    return ModelConfig(judge_provider="fake", judge_model="fake-model", temperature=0.0)


def _design_provenance():
    return compute_design_provenance(
        prompt_path=PROMPT_V12_PATH, rubric_path=RUBRIC_V12_PATH, schema_path=SCHEMA_PATH,
        gold_path=GOLD_V13_PATH, evaluation_gold_version="v1.3",
        judge_prompt_version="v1.2", judge_rubric_version="v1.2",
    )


def _row(id, question, gold_answer, claims, answerable=True):
    return Row(
        id=id, question=question, answerable=answerable, question_type="lookup",
        question_family_id="fam", policy_unit_id="pol", experiment_split="pilot",
        gold_answer=gold_answer if answerable else None,
        gold_claims=claims if answerable else [],
        unanswerable_reason=None,
    )


def _chunk(section_path, text, rank=1, doc_id="doc-x"):
    return RetrievalChunkSnapshot(
        rank=rank, chunk_id=f"c{rank}", score=0.9, doc_id=doc_id,
        section_path=section_path, source_pages=[1], text=text,
    )


def _flat(question_id, response_text, chunks, arm=ARM_CONTROL):
    return FlattenedResponse(
        question_id=question_id, arm=arm, generated_response=response_text,
        retrieved_chunks=tuple(chunks), source_experiment_file="src.json", source_record_index=0,
    )


def _completeness_only_output(label, missing=None):
    return json.dumps(
        {
            "task_completion": {"label": "PASS", "reason": "attempted"},
            "correctness": {"label": "PASS", "unverifiable_claims": [], "reason": "ok"},
            "completeness": {
                "label": label, "missing_claims": missing or [],
                "rubric_or_gold_conflict": False, "reason": "per v1.2 policy",
            },
            "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "ok"},
            "citation_support": {"label": "PASS", "unsupported_or_uncited_claims": [], "reason": "ok"},
        }
    )


# --------------------------------------------------------------------------- #
# 1. q002-style: numeric value omitted -> Completeness FAIL is a VALID,
#    schema-conforming outcome the pipeline correctly records.
# --------------------------------------------------------------------------- #
def test_q002_style_numeric_omission_documented_in_policy():
    text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    assert "MATERIAL NUMERIC DETAILS CANNOT BE OMITTED" in text
    assert "162 units" in text  # the worked example from the adjudication


def test_q002_style_completeness_fail_is_recorded_correctly(prompt_variants_v12, schema_doc):
    row = _row(
        "q002x", "What are the MISM graduation requirements?",
        "162 total units over three semesters.",
        [GoldClaim(
            claim_id="q002x-c01", claim="MISM students complete 162 total units in three semesters.",
            supporting_evidence=[SupportingEvidence(section_id="4", evidence_quote="162 units")],
        )],
    )
    flat = _flat("q002x", "You must meet the minimum total number of units taken over three semesters.",
                 [_chunk("4. Curriculum", "Students will complete 162 units in three semesters.")])
    subject = build_evaluation_subject(flat, row)
    client = FakeJudgeClient([_completeness_only_output("FAIL", missing=["q002x-c01"])])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    assert result.parsed_judge_output["completeness"]["label"] == "FAIL"


# --------------------------------------------------------------------------- #
# 2 & 3. q010-style necessary entailment: A (total alone -> not entailed,
# Completeness FAIL) vs B (total + per-period cap -> entailed, Completeness
# PASS). Verify both the policy text AND that the rendered pipeline exposes
# the extra premise Treatment has and Control lacks.
# --------------------------------------------------------------------------- #
def _q010_gold_row():
    return _row(
        "q010x", "What is the standard semester course load?",
        "MISM students typically take 54 units per semester.",
        [GoldClaim(
            claim_id="q010x-c01", claim="MISM students typically take 54 units per semester.",
            supporting_evidence=[SupportingEvidence(section_id="8.1", evidence_quote="54 units")],
        )],
    )


def test_case_a_total_alone_does_not_entail_54_each_documented():
    text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    assert "100+30+32" in text or "100 + 30 + 32" in text  # the exact counter-example
    assert "NOT necessary (a possible inference is insufficient)" in text


def test_case_b_total_plus_cap_entails_54_each_documented():
    text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    assert "no semester may exceed 54 units" in text
    assert "forces every period to equal exactly 54" in text


def test_q010_control_style_rendered_input_lacks_the_cap_premise(prompt_variants_v12):
    row = _q010_gold_row()
    flat = _flat(
        "q010x",
        "You will complete the MISM curriculum in three semesters, totaling 162 units.",
        [_chunk("4. Curriculum", "Students will complete 162 units in three semesters.")],
    )
    subject = build_evaluation_subject(flat, row)
    rendered = render_judge_request(subject, prompt_variants_v12)
    assert "162 units" in rendered.user
    assert "exceed 54" not in rendered.user  # Control never states the cap


def test_q010_treatment_style_rendered_input_has_both_premises(prompt_variants_v12):
    """This is the deterministic, offline-verifiable half of the
    calibration: the pipeline must actually expose BOTH facts a judge needs
    to apply the necessary-entailment rule, in the same rendered input."""
    row = _q010_gold_row()
    flat = _flat(
        "q010x",
        "You will complete the MISM curriculum in three semesters, totaling 162 units. "
        "The maximum units taken each semester should not exceed 54.",
        [
            _chunk("4. Curriculum", "Students will complete 162 units in three semesters.", rank=1),
            _chunk("7. Concentrations", "the maximum units taken each semester should not exceed 54",
                   rank=2),
        ],
    )
    subject = build_evaluation_subject(flat, row)
    rendered = render_judge_request(subject, prompt_variants_v12)
    assert "162 units" in rendered.user
    assert "should not exceed 54" in rendered.user


def test_q010_control_style_completeness_fail_recorded(prompt_variants_v12, schema_doc):
    row = _q010_gold_row()
    flat = _flat("q010x", "Total 162 units over three semesters.",
                 [_chunk("4. Curriculum", "Students will complete 162 units in three semesters.")])
    subject = build_evaluation_subject(flat, row)
    client = FakeJudgeClient([_completeness_only_output("FAIL", missing=["q010x-c01"])])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.parsed_judge_output["completeness"]["label"] == "FAIL"


def test_q010_treatment_style_completeness_pass_recorded(prompt_variants_v12, schema_doc):
    row = _q010_gold_row()
    flat = _flat(
        "q010x",
        "Total 162 units over three semesters; no semester may exceed 54 units.",
        [_chunk("4. Curriculum", "162 units in three semesters."),
         _chunk("7. Concentrations", "should not exceed 54", rank=2)],
    )
    subject = build_evaluation_subject(flat, row)
    client = FakeJudgeClient([_completeness_only_output("PASS")])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.parsed_judge_output["completeness"]["label"] == "PASS"


# --------------------------------------------------------------------------- #
# 4. q012-style: the REVISED v1.3 gold itself (not judge reasoning) now
# requires only ONE claim -- deterministically verifiable from the gold file.
# --------------------------------------------------------------------------- #
def test_q012_style_revised_gold_has_exactly_two_required_claims(prompt_variants_v12):
    """v1.3 q012: TWO required claims -- q012-c01 (may count) and q012-c03
    (Program Director approval required). q012-c02 ("check before
    registering") is not a required gold claim. Kept as two claims (not
    merged into one) specifically so the diagnostic distinction below
    holds: a response stating only q012-c01 is incomplete."""
    rows = load_dataset(GOLD_V13_PATH)
    gold_index = build_gold_index(rows)
    q012 = gold_index["q012"]
    claim_ids = [c.claim_id for c in q012.gold_claims]
    assert claim_ids == ["q012-c01", "q012-c03"]
    assert "q012-c02" not in claim_ids

    c01 = next(c for c in q012.gold_claims if c.claim_id == "q012-c01")
    c03 = next(c for c in q012.gold_claims if c.claim_id == "q012-c03")
    assert "may satisfy" in c01.claim
    assert "Program Director approval" in c03.claim or "official Program Director" in c03.claim

    flat = _flat(
        "q012",
        "Yes, a graduate-level course from another CMU department may satisfy the Analytic "
        'elective requirement, but official approval from the Program Director is required '
        '(see "4.2. Analytic Elective Course").',
        [_chunk("4.2. Analytic Elective Course",
                "Official approval from the Program Director is required for any course not listed below.")],
    )
    subject = build_evaluation_subject(flat, q012)
    rendered = render_judge_request(subject, prompt_variants_v12)
    assert "q012-c02" not in rendered.user
    assert "q012-c01" in rendered.user
    assert "q012-c03" in rendered.user


def test_q012_style_response_with_only_c01_is_incomplete(prompt_variants_v12, schema_doc):
    """A response stating only that the course may count (q012-c01) but
    omitting the approval condition (q012-c03) must be a representable,
    schema-valid Completeness FAIL -- this is exactly the diagnostic
    granularity the 2-claim structure exists to preserve."""
    rows = load_dataset(GOLD_V13_PATH)
    gold_index = build_gold_index(rows)
    q012 = gold_index["q012"]
    flat = _flat(
        "q012",
        "Yes, an outside-department graduate course may count.",
        [_chunk("4.2. Analytic Elective Course",
                "other Carnegie Mellon departments offer graduate-level courses that may satisfy this requirement.")],
    )
    subject = build_evaluation_subject(flat, q012)
    client = FakeJudgeClient([_completeness_only_output("FAIL", missing=["q012-c03"])])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    assert result.parsed_judge_output["completeness"]["label"] == "FAIL"
    assert result.parsed_judge_output["completeness"]["missing_claims"] == ["q012-c03"]


def test_q012_style_completeness_pass_recorded_under_revised_gold(prompt_variants_v12, schema_doc):
    rows = load_dataset(GOLD_V13_PATH)
    gold_index = build_gold_index(rows)
    q012 = gold_index["q012"]
    flat = _flat(
        "q012",
        "Yes, a graduate-level course from another CMU department may satisfy the Analytic "
        "elective requirement, but official approval from the Program Director is required.",
        [_chunk("4.2. Analytic Elective Course", "Official approval from the Program Director is required.")],
    )
    subject = build_evaluation_subject(flat, q012)
    client = FakeJudgeClient([_completeness_only_output("PASS")])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.parsed_judge_output["completeness"]["label"] == "PASS"


# --------------------------------------------------------------------------- #
# 5 & 6. q019-style: deterministic arithmetic derivation allowed; two
# different quantities are not a contradiction merely because they differ.
# --------------------------------------------------------------------------- #
def test_case_5_deterministic_derivation_documented():
    text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    assert "36-6-6-6=18" in text or "36 − 6 − 6 − 6 = 18" in text or "36-6-6-6" in text


def test_case_6_quantity_identity_documented():
    text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    assert "NUMERIC-QUANTITY IDENTITY" in text
    assert "18 != 6" in text or "18 additional units" in text


def test_q019_style_pipeline_exposes_all_arithmetic_premises_in_citation_scope(prompt_variants_v12):
    row = _row(
        "q019x", "How many additional units are needed if I choose Demystifying AI?",
        "18 additional units.",
        [GoldClaim(
            claim_id="q019x-c01", claim="18 additional units are needed.",
            supporting_evidence=[SupportingEvidence(section_id="7.1.1", evidence_quote="36 units")],
        )],
    )
    chunk_text = (
        "### 7.1.1. AI Management Requirements\n\n"
        "Students must complete 36 units of the following: Fundamentals of Operationalizing AI "
        "(6 units), Responsible AI (6 units), Demystifying AI (6 units) OR Introduction to AI "
        "(12 units), and at least 12-18 units from the elective list."
    )
    flat = _flat(
        "q019x",
        'If you choose Demystifying AI, you need 18 additional units (see "7.1.1. AI Management Requirements").',
        [_chunk("7.1.1. AI Management Requirements", chunk_text)],
    )
    subject = build_evaluation_subject(flat, row)
    res = subject.citation_resolutions[0]
    assert res.resolution_status == "RESOLVED"
    # every arithmetic premise (36, two 6s, the 6-or-12 alternative) is
    # inside the citation's own allowed evidence scope -- deterministically
    # verifiable without any model call.
    for premise in ("36 units", "6 units) OR", "12 units)"):
        assert premise in res.resolved_scope_retrieved_text


def test_q019_style_correctness_faithfulness_citation_support_pass_recorded(
    prompt_variants_v12, schema_doc
):
    row = _row(
        "q019x", "How many additional units are needed if I choose Demystifying AI?",
        "18 additional units.",
        [GoldClaim(
            claim_id="q019x-c01", claim="18 additional units are needed.",
            supporting_evidence=[SupportingEvidence(section_id="7.1.1", evidence_quote="36 units")],
        )],
    )
    flat = _flat(
        "q019x",
        'If you choose Demystifying AI (6 units) instead of Introduction to AI (12 units), you '
        "need 18 additional units to reach the 36-unit requirement. The two options differ by 6 "
        'units (see "7.1.1. AI Management Requirements").',
        [_chunk("7.1.1. AI Management Requirements",
                "36 units total. Demystifying AI (6 units) OR Introduction to AI (12 units).")],
    )
    subject = build_evaluation_subject(flat, row)
    output = json.dumps(
        {
            "task_completion": {"label": "PASS", "reason": "attempted"},
            "correctness": {"label": "PASS", "unverifiable_claims": [], "reason": "deterministic derivation"},
            "completeness": {"label": "PASS", "missing_claims": [], "rubric_or_gold_conflict": False, "reason": "ok"},
            "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "18=36-6-6-6, all premises present"},
            "citation_support": {"label": "PASS", "unsupported_or_uncited_claims": [], "reason": "within scope"},
        }
    )
    client = FakeJudgeClient([output])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    assert result.parsed_judge_output["correctness"]["label"] == "PASS"
    assert result.parsed_judge_output["faithfulness"]["label"] == "PASS"
    assert result.parsed_judge_output["citation_support"]["label"] == "PASS"


# --------------------------------------------------------------------------- #
# 7. Correctness fallback: extra claim absent from gold, present in
# retrieved context -> must NOT be forced to REVIEW_REQUIRED.
# --------------------------------------------------------------------------- #
def test_case_7_correctness_evidence_policy_documented():
    text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    assert "CORRECTNESS EVIDENCE POLICY" in text
    assert "you MAY additionally" in text.replace("\n", " ")


def test_case_7_gold_and_retrieved_context_are_both_in_the_same_rendered_message(prompt_variants_v12):
    """The fallback policy is only usable if the judge actually receives
    retrieved_context alongside the gold reference material in one call --
    verify the answerable template structurally guarantees this."""
    row = _row(
        "q010y", "q?", "gold answer text",
        [GoldClaim(claim_id="q010y-c01", claim="a claim", supporting_evidence=[
            SupportingEvidence(section_id="1", evidence_quote="q")])],
    )
    flat = _flat("q010y", "a response",
                 [_chunk("7. Concentrations", "should not exceed 54 UNIQUE_RETRIEVED_MARKER")])
    subject = build_evaluation_subject(flat, row)
    rendered = render_judge_request(subject, prompt_variants_v12)
    assert "GOLD REFERENCE MATERIAL" in rendered.user
    assert "UNIQUE_RETRIEVED_MARKER" in rendered.user  # retrieved context is in the SAME message


def test_case_7_correctness_pass_for_gold_silent_retrieved_supported_claim_recorded(
    prompt_variants_v12, schema_doc
):
    row = _row(
        "q010z", "What is the semester course load?", "162 units over three semesters.",
        [GoldClaim(claim_id="q010z-c01", claim="162 units over three semesters", supporting_evidence=[
            SupportingEvidence(section_id="4", evidence_quote="162 units")])],
    )
    flat = _flat(
        "q010z",
        "162 units over three semesters. The maximum units taken each semester should not exceed 54.",
        [_chunk("4. Curriculum", "162 units in three semesters."),
         _chunk("7. Concentrations", "maximum units taken each semester should not exceed 54", rank=2)],
    )
    subject = build_evaluation_subject(flat, row)
    output = json.dumps(
        {
            "task_completion": {"label": "PASS", "reason": "ok"},
            "correctness": {
                "label": "PASS", "unverifiable_claims": [],
                "reason": "the 54-unit cap is absent from gold but directly supported by retrieved context",
            },
            "completeness": {"label": "PASS", "missing_claims": [], "rubric_or_gold_conflict": False, "reason": "ok"},
            "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "ok"},
            "citation_support": {"label": "PASS", "unsupported_or_uncited_claims": [], "reason": "ok"},
        }
    )
    client = FakeJudgeClient([output])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    assert result.parsed_judge_output["correctness"]["label"] == "PASS"


# --------------------------------------------------------------------------- #
# 8. No-evidence case: claim absent from BOTH gold and retrieved context ->
# REVIEW_REQUIRED remains a valid, schema-conforming outcome.
# --------------------------------------------------------------------------- #
def test_case_8_review_required_remains_available_and_valid(schema_doc):
    output = {
        "task_completion": {"label": "PASS", "reason": "ok"},
        "correctness": {
            "label": "REVIEW_REQUIRED",
            "unverifiable_claims": ["a claim absent from both gold and retrieved context"],
            "reason": "neither gold nor retrieved context verifies or contradicts this claim",
        },
        "completeness": {"label": "PASS", "missing_claims": [], "rubric_or_gold_conflict": False, "reason": "ok"},
        "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "ok"},
        "citation_support": {"label": "PASS", "unsupported_or_uncited_claims": [], "reason": "ok"},
    }
    errors = validate_judge_output(output, answerable=True, schema_doc=schema_doc)
    assert errors == []


def test_case_8_review_required_end_to_end_recorded_not_coerced(prompt_variants_v12, schema_doc):
    row = _row(
        "q010w", "q?", "gold answer",
        [GoldClaim(claim_id="q010w-c01", claim="a claim", supporting_evidence=[
            SupportingEvidence(section_id="1", evidence_quote="q")])],
    )
    flat = _flat("q010w", "a response with an unverifiable extra claim",
                 [_chunk("4. Curriculum", "unrelated retrieved text")])
    subject = build_evaluation_subject(flat, row)
    output = json.dumps(
        {
            "task_completion": {"label": "PASS", "reason": "ok"},
            "correctness": {
                "label": "REVIEW_REQUIRED", "unverifiable_claims": ["extra claim"],
                "reason": "not verifiable from gold or retrieved context",
            },
            "completeness": {"label": "PASS", "missing_claims": [], "rubric_or_gold_conflict": False, "reason": "ok"},
            "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "ok"},
            "citation_support": {"label": "PASS", "unsupported_or_uncited_claims": [], "reason": "ok"},
        }
    )
    client = FakeJudgeClient([output])
    result = evaluate_subject(
        subject, prompt_variants_v12, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    # REVIEW_REQUIRED is preserved verbatim, never silently coerced to PASS/FAIL.
    assert result.parsed_judge_output["correctness"]["label"] == "REVIEW_REQUIRED"
    assert result.parsed_judge_output["correctness"]["unverifiable_claims"] == ["extra claim"]


# --------------------------------------------------------------------------- #
# Citation infrastructure frozen: v1.2 (citation + semantic) carries forward
# v1.1's (citation-only) hierarchy text completely unchanged -- semantic
# calibration touched only Correctness/Completeness/Faithfulness/one
# Citation Support addendum, never resolution/hierarchy/scope itself.
# --------------------------------------------------------------------------- #
PROMPT_V11_PATH = "eval/judge/judge_prompt_v1.1.txt"
RUBRIC_V11_PATH = "eval/judge/judge_rubric_v1.1.md"


def test_v12_prompt_carries_forward_v11_citation_hierarchy_text_unchanged():
    v11_text = open(PROMPT_V11_PATH, encoding="utf-8").read()
    v12_text = open(PROMPT_V12_PATH, encoding="utf-8").read()
    marker_start = "THREE CASES YOU MAY ENCOUNTER"
    marker_end = "--- INPUT TEMPLATE ---"
    assert v11_text[v11_text.index(marker_start):v11_text.index(marker_end)] == \
        v12_text[v12_text.index(marker_start):v12_text.index(marker_end)]


def test_v12_rubric_hierarchy_and_citation_validity_sections_match_v11():
    v11_text = open(RUBRIC_V11_PATH, encoding="utf-8").read()
    v12_text = open(RUBRIC_V12_PATH, encoding="utf-8").read()
    marker_start = "**Hierarchy-aware evidence scope (adopted policy).**"
    marker_end = "## 7. Citation Validity"
    assert v11_text[v11_text.index(marker_start):v11_text.index(marker_end)] == \
        v12_text[v12_text.index(marker_start):v12_text.index(marker_end)]
    sec7_start = "## 7. Citation Validity"
    sec7_end = "## 8. Correct Abstention"
    assert v11_text[v11_text.index(sec7_start):v11_text.index(sec7_end)] == \
        v12_text[v12_text.index(sec7_start):v12_text.index(sec7_end)]
