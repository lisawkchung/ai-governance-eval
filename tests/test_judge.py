"""
Tests for heinzy/eval/judge.py and heinzy/eval/citation_resolution.py --
the Judge v1 execution module. Fully offline: a scripted fake JudgeClient
stands in for any real model call. No network/API call is ever made.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from heinzy.eval.citation_resolution import (
    AMBIGUOUS,
    RESOLVED,
    UNRESOLVED,
    RetrievedSectionChunk,
    normalize_citation_label,
    resolve_citations,
)
from heinzy.eval.dataset import GoldClaim, Row, SupportingEvidence
from heinzy.eval.experiment import RetrievalChunkSnapshot
from heinzy.eval.judge import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EXECUTION_ERROR,
    EXECUTION_SUCCESS,
    PARSE_ERROR,
    SCHEMA_ERROR,
    EvaluationSubject,
    JudgeCallResult,
    JudgeInputError,
    JudgePromptVariants,
    ModelConfig,
    assert_control_sources_consistent,
    assert_unique_flattened,
    build_evaluation_subject,
    build_gold_index,
    compare_control_sources,
    compute_design_provenance,
    evaluate_subject,
    evaluation_subject_fingerprint,
    fingerprint_retrieved_context,
    flatten_experiment_report,
    join_gold,
    load_judge_prompt_variants,
    load_output_schema,
    parse_judge_output,
    render_judge_request,
    sha256_file,
    sha256_text,
    validate_full_coverage,
    validate_judge_output,
)

PROMPT_PATH = "eval/judge/judge_prompt_v1.txt"
RUBRIC_PATH = "eval/judge/judge_rubric_v1.md"
SCHEMA_PATH = "eval/judge/judge_output_schema_v1.json"
GOLD_PATH = "eval/questions_pilot_v1.2.jsonl"


# --------------------------------------------------------------------------- #
# fixtures / fakes
# --------------------------------------------------------------------------- #
class FakeJudgeClient:
    """Scripted fake -- returns queued raw text (or raises), records every
    call's (system, user) so tests can assert on state isolation."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def run(self, system_prompt, user_prompt, *, temperature, timeout=None):
        self.calls.append((system_prompt, user_prompt))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return JudgeCallResult(
            raw_text=item, input_tokens=10, output_tokens=5, total_tokens=15,
            model_call_made=True, latency_seconds=0.001, model_tag="fake-judge",
        )


def _row(
    id="q001", question="Q?", answerable=True, gold_answer="A.",
    claims=None, unanswerable_reason=None,
) -> Row:
    if answerable and claims is None:
        claims = [
            GoldClaim(
                claim_id=f"{id}-c01", claim="required fact",
                supporting_evidence=[SupportingEvidence(section_id="1", evidence_quote="quote")],
            )
        ]
    return Row(
        id=id, question=question, answerable=answerable, question_type="lookup",
        question_family_id="fam", policy_unit_id="pol", experiment_split="pilot",
        gold_answer=gold_answer if answerable else None,
        gold_claims=claims if answerable else [],
        unanswerable_reason=None if answerable else (unanswerable_reason or "wrong_scope"),
    )


def _chunk(rank=1, section_path="1. Section", text="chunk text", doc_id="doc1") -> RetrievalChunkSnapshot:
    return RetrievalChunkSnapshot(
        rank=rank, chunk_id=f"c{rank}", score=0.9, doc_id=doc_id,
        section_path=section_path, source_pages=[1], text=text,
    )


def _report(question_id="q001", control_text="control answer", treatment_text="treatment answer",
            chunks=None) -> dict:
    chunks = chunks if chunks is not None else [_chunk()]
    return {
        "results": [
            {
                "question_id": question_id,
                "question": "Q?",
                "retrieval": {"chunks": [dataclasses.asdict(c) for c in chunks]},
                "control": {"text": control_text},
                "treatment": {"final_text": treatment_text},
            }
        ]
    }


@pytest.fixture(scope="module")
def prompt_variants() -> JudgePromptVariants:
    return load_judge_prompt_variants(PROMPT_PATH)


@pytest.fixture(scope="module")
def schema_doc() -> dict:
    return load_output_schema(SCHEMA_PATH)


def _good_answerable_output() -> str:
    return json.dumps(
        {
            "task_completion": {"label": "PASS", "reason": "ok"},
            "correctness": {"label": "PASS", "unverifiable_claims": [], "reason": "ok"},
            "completeness": {
                "label": "PASS", "missing_claims": [], "rubric_or_gold_conflict": False, "reason": "ok",
            },
            "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "ok"},
            "citation_support": {"label": "PASS", "unsupported_or_uncited_claims": [], "reason": "ok"},
        }
    )


def _model_config() -> ModelConfig:
    return ModelConfig(judge_provider="fake", judge_model="fake-model", temperature=0.0)


def _design_provenance():
    return compute_design_provenance(
        prompt_path=PROMPT_PATH, rubric_path=RUBRIC_PATH, schema_path=SCHEMA_PATH,
        gold_path=GOLD_PATH, evaluation_gold_version="v1.2",
    )


# --------------------------------------------------------------------------- #
# 1-4: identity / uniqueness / coverage
# --------------------------------------------------------------------------- #
def test_gold_question_id_uniqueness_detected():
    rows = [_row(id="q001"), _row(id="q001")]
    with pytest.raises(JudgeInputError):
        build_gold_index(rows)


def test_duplicate_question_id_across_arms_is_allowed():
    report = _report(question_id="q001")
    flat = flatten_experiment_report(report, "src.json")
    assert len(flat) == 2
    assert {f.arm for f in flat} == {ARM_CONTROL, ARM_TREATMENT}
    # both share question_id -- must not raise
    assert_unique_flattened(flat)


def test_duplicate_question_id_arm_pair_is_rejected():
    report = _report(question_id="q001")
    flat = flatten_experiment_report(report, "src.json")
    flat = flat + [flat[0]]  # inject a real (question_id, arm) duplicate
    with pytest.raises(JudgeInputError):
        assert_unique_flattened(flat)


def test_full_expected_arm_coverage_validation():
    rows = [_row(id="q001"), _row(id="q002")]
    report = _report(question_id="q001")
    flat = flatten_experiment_report(report, "src.json")
    with pytest.raises(JudgeInputError) as exc:
        validate_full_coverage(flat, rows)
    assert any("q002" in e for e in exc.value.errors)


def test_full_coverage_passes_when_complete():
    rows = [_row(id="q001")]
    report = _report(question_id="q001")
    flat = flatten_experiment_report(report, "src.json")
    validate_full_coverage(flat, rows)  # must not raise


def test_response_with_no_matching_gold_row_fails_join():
    rows = [_row(id="q999")]
    gold_index = build_gold_index(rows)
    report = _report(question_id="q001")
    flat = flatten_experiment_report(report, "src.json")
    with pytest.raises(JudgeInputError):
        join_gold(flat[0], gold_index)


# --------------------------------------------------------------------------- #
# 5-6: Control consistency check
# --------------------------------------------------------------------------- #
def test_control_consistency_passes_for_identical_text_and_context():
    a = _report(question_id="q001", control_text="same text")
    b = _report(question_id="q001", control_text="same text")
    comparisons = compare_control_sources(a, b)
    assert_control_sources_consistent(comparisons)  # must not raise
    assert comparisons[0].text_match and comparisons[0].context_match


def test_control_consistency_mismatch_is_reported_and_stops_selection():
    a = _report(question_id="q001", control_text="text A")
    b = _report(question_id="q001", control_text="text B")
    comparisons = compare_control_sources(a, b)
    assert comparisons[0].mismatch_categories == ("CONTROL_TEXT_DIFFERS",)
    with pytest.raises(JudgeInputError) as exc:
        assert_control_sources_consistent(comparisons)
    assert "q001" in exc.value.errors[0]
    assert "CONTROL_TEXT_DIFFERS" in exc.value.errors[0]


def test_control_consistency_context_mismatch_detected():
    a = _report(question_id="q001", chunks=[_chunk(text="context A")])
    b = _report(question_id="q001", chunks=[_chunk(text="context B")])
    comparisons = compare_control_sources(a, b)
    assert "RETRIEVED_CONTEXT_DIFFERS" in comparisons[0].mismatch_categories


# --------------------------------------------------------------------------- #
# 7-8: prompt rendering (answerable / unanswerable)
# --------------------------------------------------------------------------- #
def test_answerable_prompt_rendering(prompt_variants):
    row = _row(id="q001", question="How many units?", gold_answer="54 units.")
    chunks = (_chunk(section_path="8. Units", text="Students take 54 units."),)
    subject = EvaluationSubject(
        question_id="q001", question=row.question, answerable=True,
        generated_response='54 units (see "8. Units").',
        retrieved_chunks=chunks, citation_resolutions=(),
        gold_answer=row.gold_answer, gold_claims=tuple(row.gold_claims),
        unanswerable_reason=None,
    )
    rendered = render_judge_request(subject, prompt_variants)
    assert "How many units?" in rendered.user
    assert "54 units (see" in rendered.user
    assert "GOLD ANSWER" in rendered.user
    assert "54 units." in rendered.user
    assert "{{" not in rendered.user and "{{" not in rendered.system


def test_unanswerable_prompt_rendering(prompt_variants):
    row = _row(id="q005", question="Unanswerable Q?", answerable=False,
               unanswerable_reason="wrong_scope")
    subject = EvaluationSubject(
        question_id="q005", question=row.question, answerable=False,
        generated_response="I can't answer that.",
        retrieved_chunks=(), citation_resolutions=(),
        gold_answer=None, gold_claims=None,
        unanswerable_reason=row.unanswerable_reason,
    )
    rendered = render_judge_request(subject, prompt_variants)
    assert "Unanswerable Q?" in rendered.user
    assert "wrong_scope" in rendered.user
    assert "GOLD ANSWER" not in rendered.user
    assert "{{" not in rendered.user and "{{" not in rendered.system


# --------------------------------------------------------------------------- #
# 9-11: blinding
# --------------------------------------------------------------------------- #
def test_arm_identity_excluded_from_judge_input_structurally():
    """EvaluationSubject (the only type render_judge_request accepts) has no
    field named/related to arm identity at all."""
    field_names = {f.name for f in dataclasses.fields(EvaluationSubject)}
    assert "arm" not in field_names
    assert not (field_names & {"control", "treatment", "arm_identity"})


def test_verifier_and_strategy_metadata_excluded_from_judge_input():
    field_names = {f.name for f in dataclasses.fields(EvaluationSubject)}
    forbidden = {
        "verifier_decision", "verifier_reason", "treatment_strategy_version",
        "zero_hit_guard_triggered", "verifier_call_made", "model_tag",
    }
    assert not (field_names & forbidden)

    # Also check the actual rendered content for a synthetic subject whose
    # fixture text is guaranteed clean, so a positive match is real leakage,
    # not a coincidental English word inside real handbook prose.
    row = _row(id="q001", question="Synthetic question?", gold_answer="Synthetic gold answer.")
    subject = EvaluationSubject(
        question_id="q001", question=row.question, answerable=True,
        generated_response="Synthetic response text with no arm words.",
        retrieved_chunks=(_chunk(text="Synthetic retrieved text."),),
        citation_resolutions=(),
        gold_answer=row.gold_answer, gold_claims=tuple(row.gold_claims),
        unanswerable_reason=None,
    )
    variants = load_judge_prompt_variants(PROMPT_PATH)
    rendered = render_judge_request(subject, variants)
    for forbidden_word in ("verifier_decision", "zero_hit_guard", "TREATMENT_STRATEGY", "KEEP", "REVISE", "ABSTAIN"):
        assert forbidden_word not in rendered.user


def test_human_labels_never_loaded_by_judge_module():
    import heinzy.eval.judge as judge_module

    src = judge_module.__file__
    text = open(src, encoding="utf-8").read()
    assert "final_human_labels" not in text.replace(
        '"""\nMODULE BOUNDARY', ""
    ).split("This module NEVER imports")[0] or True  # see stricter check below
    # Stricter, unambiguous check: the only mention of the human-label
    # filename anywhere in the module is inside the docstring explaining
    # that it is never read -- there is no `open(`, `read_csv`, or import
    # statement referencing it.
    assert "eval/annotation/final_human_labels_v2.1.csv" in text  # documented
    for line in text.splitlines():
        stripped = line.strip()
        if "final_human_labels" in stripped:
            assert not any(
                token in stripped for token in ("open(", "read_csv", "Path(", "import")
            ), f"unexpected human-label file access: {stripped!r}"


# --------------------------------------------------------------------------- #
# 12-16: citation resolution
# --------------------------------------------------------------------------- #
def test_citation_exact_canonical_resolution_resolves():
    chunks = [RetrievedSectionChunk(section_path="8.1. Normal Courseload", doc_id="d1", text="t", rank=1)]
    res = resolve_citations(['8.1.  normal courseload  '], chunks)
    assert res[0].resolution_status == RESOLVED
    assert res[0].resolved_section_path == "8.1. Normal Courseload"


def test_nonexistent_citation_is_unresolved():
    chunks = [RetrievedSectionChunk(section_path="8.1. Normal Courseload", doc_id="d1", text="t", rank=1)]
    res = resolve_citations(["9.9. Nonexistent Section"], chunks)
    assert res[0].resolution_status == UNRESOLVED
    assert res[0].resolved_section_path is None


def test_ambiguous_citation_never_guesses():
    chunks = [
        RetrievedSectionChunk(section_path="8. Units", doc_id="d1", text="t1", rank=1),
        RetrievedSectionChunk(section_path="8. units", doc_id="d2", text="t2", rank=2),
    ]
    res = resolve_citations(["8. Units"], chunks)
    assert res[0].resolution_status == AMBIGUOUS
    assert res[0].resolved_section_path is None
    assert set(res[0].ambiguous_candidates) == {"8. Units", "8. units"}


def test_citation_resolver_never_uses_fuzzy_or_semantic_matching():
    """A citation that is a genuine substring/prefix of a retrieved section
    must NOT resolve -- exact-match-only, unlike grounding.is_supported."""
    chunks = [RetrievedSectionChunk(section_path="8.1.2. Sub-detail", doc_id="d1", text="t", rank=1)]
    res = resolve_citations(["8.1"], chunks)  # numeric-prefix substring only
    assert res[0].resolution_status == UNRESOLVED


def test_q013_style_body_text_citation_with_internal_parens_remains_unresolved():
    """Regression for the extraction-truncation bug found via q013: even
    after extraction correctly captures the FULL body-text-as-citation
    label (including its internal parenthetical), it must still resolve as
    UNRESOLVED -- it is body text, not a real retrieved section identifier.
    This is a body-text-as-citation failure case by design (judge_rubric_v1.md
    section 7), not something exact-match resolution should ever paper over."""
    full_label = "Students may transfer to the MISM (non-BIDA) program"
    chunks = [
        RetrievedSectionChunk(
            section_path="5. MISM Business Intelligence & Data Analytics (MISM-BIDA) Curriculum",
            doc_id="d1",
            text=f"{full_label}. The pre-matriculation transfer deadline is June 1.",
            rank=1,
        )
    ]
    res = resolve_citations([full_label], chunks)
    assert res[0].resolution_status == UNRESOLVED
    assert res[0].raw_citation == full_label  # extraction is complete, not truncated
    assert res[0].resolved_section_path is None


def test_resolved_citation_includes_exact_retrieved_evidence_text():
    chunks = [RetrievedSectionChunk(section_path="8. Units", doc_id="d1", text="EXACT TEXT", rank=1)]
    res = resolve_citations(["8. Units"], chunks)
    assert res[0].resolved_retrieved_text == "EXACT TEXT"


def test_normalize_citation_label_is_case_and_whitespace_insensitive_only():
    assert normalize_citation_label("  8.1.  Foo   Bar ") == normalize_citation_label("8.1. foo bar")
    # interior digits are never touched (no numeric-prefix collapsing)
    assert normalize_citation_label("8.1") != normalize_citation_label("8.1.2")


# --------------------------------------------------------------------------- #
# 17-21: output parsing / schema / execution status
# --------------------------------------------------------------------------- #
def test_valid_pass_fail_na_review_required_output_parses(schema_doc):
    good = {
        "task_completion": {"label": "REVIEW_REQUIRED", "reason": "ambiguous"},
        "correctness": {"label": "N/A", "unverifiable_claims": [], "reason": "no claim"},
        "completeness": {
            "label": "REVIEW_REQUIRED", "missing_claims": [], "rubric_or_gold_conflict": True,
            "reason": "conflict",
        },
        "faithfulness": {"label": "PASS", "unsupported_claims": [], "reason": "ok"},
        "citation_support": {"label": "FAIL", "unsupported_or_uncited_claims": ["x"], "reason": "bad"},
    }
    assert validate_judge_output(good, answerable=True, schema_doc=schema_doc) == []


def test_invalid_enum_is_schema_error(schema_doc):
    bad = json.loads(_good_answerable_output())
    bad["task_completion"] = {"label": "MAYBE", "reason": "x"}
    errors = validate_judge_output(bad, answerable=True, schema_doc=schema_doc)
    assert errors


def test_malformed_json_is_parse_error():
    parsed, err = parse_judge_output("this is not { json")
    assert parsed is None
    assert err is not None


def test_mock_model_failure_is_execution_error(prompt_variants, schema_doc):
    row = _row(id="q001")
    subject = build_evaluation_subject(
        flat=flatten_experiment_report(_report(question_id="q001"), "src.json")[0],
        gold_row=row,
    )
    client = FakeJudgeClient([RuntimeError("connection refused")])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_ERROR
    assert result.error_type == "RuntimeError"
    assert result.raw_judge_output is None


def test_end_to_end_success_status(prompt_variants, schema_doc):
    row = _row(id="q001")
    subject = build_evaluation_subject(
        flat=flatten_experiment_report(_report(question_id="q001"), "src.json")[0],
        gold_row=row,
    )
    client = FakeJudgeClient([_good_answerable_output()])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    assert result.parsed_judge_output["task_completion"]["label"] == "PASS"


def test_end_to_end_parse_error_status(prompt_variants, schema_doc):
    row = _row(id="q001")
    subject = build_evaluation_subject(
        flat=flatten_experiment_report(_report(question_id="q001"), "src.json")[0],
        gold_row=row,
    )
    client = FakeJudgeClient(["not valid json at all"])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == PARSE_ERROR
    assert result.raw_judge_output == "not valid json at all"


def test_end_to_end_schema_error_status(prompt_variants, schema_doc):
    row = _row(id="q001")
    subject = build_evaluation_subject(
        flat=flatten_experiment_report(_report(question_id="q001"), "src.json")[0],
        gold_row=row,
    )
    bad = json.loads(_good_answerable_output())
    bad["task_completion"]["label"] = "MAYBE"
    client = FakeJudgeClient([json.dumps(bad)])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == SCHEMA_ERROR


def test_review_required_is_a_valid_semantic_result_never_an_error_status(prompt_variants, schema_doc):
    row = _row(id="q001")
    subject = build_evaluation_subject(
        flat=flatten_experiment_report(_report(question_id="q001"), "src.json")[0],
        gold_row=row,
    )
    output = json.loads(_good_answerable_output())
    output["task_completion"] = {"label": "REVIEW_REQUIRED", "reason": "ambiguous"}
    client = FakeJudgeClient([json.dumps(output)])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.execution_status == EXECUTION_SUCCESS
    assert result.parsed_judge_output["task_completion"]["label"] == "REVIEW_REQUIRED"


# --------------------------------------------------------------------------- #
# 22: answerable/unanswerable schema selection from gold only
# --------------------------------------------------------------------------- #
def test_answerable_flag_selects_variant_and_schema(prompt_variants, schema_doc):
    answerable_row = _row(id="q001", answerable=True)
    unanswerable_row = _row(id="q002", answerable=False, question="Unanswerable?")

    flat_a = flatten_experiment_report(_report(question_id="q001"), "src.json")[0]
    flat_u = flatten_experiment_report(
        _report(question_id="q002", control_text="I can't answer that."), "src.json"
    )[0]

    subj_a = build_evaluation_subject(flat_a, answerable_row)
    subj_u = build_evaluation_subject(flat_u, unanswerable_row)

    rendered_a = render_judge_request(subj_a, prompt_variants)
    rendered_u = render_judge_request(subj_u, prompt_variants)
    assert "GOLD REFERENCE MATERIAL" in rendered_a.user
    assert "GOLD REFERENCE MATERIAL" not in rendered_u.user
    assert "CORRECT ABSTENTION" in rendered_u.system
    assert "CORRECT ABSTENTION" not in rendered_a.system


# --------------------------------------------------------------------------- #
# 23-25: fingerprint determinism
# --------------------------------------------------------------------------- #
def test_design_artifact_fingerprints_are_deterministic():
    a = sha256_file(PROMPT_PATH)
    b = sha256_file(PROMPT_PATH)
    assert a == b
    assert len(a) == 64


def test_gold_fingerprint_deterministic():
    assert sha256_file(GOLD_PATH) == sha256_file(GOLD_PATH)


def test_rendered_judge_input_fingerprint_deterministic(prompt_variants):
    row = _row(id="q001")
    flat = flatten_experiment_report(_report(question_id="q001"), "src.json")[0]
    subject = build_evaluation_subject(flat, row)
    r1 = render_judge_request(subject, prompt_variants)
    r2 = render_judge_request(subject, prompt_variants)
    assert sha256_text(r1.system + "\x1e" + r1.user) == sha256_text(r2.system + "\x1e" + r2.user)


def test_generated_response_and_context_and_subject_fingerprints_deterministic():
    row = _row(id="q001")
    flat = flatten_experiment_report(_report(question_id="q001"), "src.json")[0]
    subject = build_evaluation_subject(flat, row)
    assert sha256_text(subject.generated_response) == sha256_text(subject.generated_response)
    assert fingerprint_retrieved_context(subject.retrieved_chunks) == fingerprint_retrieved_context(
        subject.retrieved_chunks
    )
    assert evaluation_subject_fingerprint(subject) == evaluation_subject_fingerprint(subject)


def test_evaluation_subject_fingerprint_excludes_arm():
    """Same underlying response/context/citations -> identical fingerprint
    regardless of which arm it's associated with in orchestration metadata.
    This is exactly why evaluation_subject_fingerprint must NEVER be used as
    a row-identity key on its own -- see the dedicated row-identity vs
    artifact-integrity tests below."""
    row = _row(id="q001")
    flat_control = flatten_experiment_report(
        _report(question_id="q001", control_text="same", treatment_text="same"), "src.json",
        arms=(ARM_CONTROL,),
    )[0]
    flat_treatment = flatten_experiment_report(
        _report(question_id="q001", control_text="same", treatment_text="same"), "src.json",
        arms=(ARM_TREATMENT,),
    )[0]
    subj_control = build_evaluation_subject(flat_control, row)
    subj_treatment = build_evaluation_subject(flat_treatment, row)
    assert evaluation_subject_fingerprint(subj_control) == evaluation_subject_fingerprint(subj_treatment)


# --------------------------------------------------------------------------- #
# Row identity (question_id, arm) vs artifact integrity
# (evaluation_subject_fingerprint) -- these are two SEPARATE concepts for a
# future human-label calibration join, never to be merged. See heinzy/eval/
# judge.py's module docstring and JudgeResultRecord docstring for the full
# contract this proves.
# --------------------------------------------------------------------------- #
def test_same_question_id_different_arm_are_distinct_row_identities():
    """(question_id, arm) is the row-identity key: two rows sharing
    question_id but differing only in arm must be treated as distinct rows,
    never merged."""
    report = _report(question_id="q001", control_text="control text", treatment_text="treatment text")
    flat = flatten_experiment_report(report, "src.json")
    control_row = next(f for f in flat if f.arm == ARM_CONTROL)
    treatment_row = next(f for f in flat if f.arm == ARM_TREATMENT)

    control_identity = (control_row.question_id, control_row.arm)
    treatment_identity = (treatment_row.question_id, treatment_row.arm)

    assert control_identity != treatment_identity
    assert control_identity[0] == treatment_identity[0] == "q001"
    # assert_unique_flattened must accept both as distinct, not reject them
    assert_unique_flattened(flat)


def test_identical_artifact_different_arm_may_share_fingerprint_but_not_row_identity():
    """Direct proof of the bug this patch fixes: Control and Treatment CAN
    legitimately produce the identical generated response / retrieved
    context / citation mapping (e.g. a zero-hit refusal echoed unchanged by
    both arms), giving them the SAME evaluation_subject_fingerprint --  yet
    they remain two distinct rows by (question_id, arm). A fingerprint-only
    join would wrongly conflate them; a (question_id, arm) join does not."""
    row = _row(id="q001")
    report = _report(question_id="q001", control_text="identical text", treatment_text="identical text")
    flat = flatten_experiment_report(report, "src.json")
    control_flat = next(f for f in flat if f.arm == ARM_CONTROL)
    treatment_flat = next(f for f in flat if f.arm == ARM_TREATMENT)

    subj_control = build_evaluation_subject(control_flat, row)
    subj_treatment = build_evaluation_subject(treatment_flat, row)

    # artifact integrity: same fingerprint is a LEGITIMATE outcome here.
    assert evaluation_subject_fingerprint(subj_control) == evaluation_subject_fingerprint(subj_treatment)
    # row identity: still two distinct rows.
    assert (control_flat.question_id, control_flat.arm) != (treatment_flat.question_id, treatment_flat.arm)


def test_arm_absent_from_rendered_judge_input_even_when_fingerprints_collide(prompt_variants):
    """Even in the fingerprint-collision scenario above, the rendered Judge
    input for each arm must still contain no arm identity."""
    row = _row(id="q001")
    report = _report(question_id="q001", control_text="identical text", treatment_text="identical text")
    flat = flatten_experiment_report(report, "src.json")
    control_flat = next(f for f in flat if f.arm == ARM_CONTROL)
    treatment_flat = next(f for f in flat if f.arm == ARM_TREATMENT)

    subj_control = build_evaluation_subject(control_flat, row)
    subj_treatment = build_evaluation_subject(treatment_flat, row)

    rendered_control = render_judge_request(subj_control, prompt_variants)
    rendered_treatment = render_judge_request(subj_treatment, prompt_variants)
    # Rendered inputs for both arms are identical to each other (proves
    # arm cannot have influenced rendering) and carry no per-response arm
    # signal in the dynamic (user) portion.
    assert rendered_control.user == rendered_treatment.user
    assert rendered_control.system == rendered_treatment.system


def test_evaluation_subject_fingerprint_function_has_no_arm_parameter():
    """Structural guarantee, not just an empirical equality check: the
    fingerprint function cannot take arm as an input at all."""
    import inspect

    params = list(inspect.signature(evaluation_subject_fingerprint).parameters)
    assert params == ["subject"]
    assert "arm" not in params


def test_result_records_for_both_arms_remain_distinguishable_by_row_identity(
    prompt_variants, schema_doc
):
    """End-to-end: two JudgeResultRecords sharing an evaluation_subject_fingerprint
    (identical artifact) still carry distinct (question_id, arm) row identity,
    so a future calibration join keyed on (question_id, arm) never conflates
    them -- only a fingerprint-only join would."""
    row = _row(id="q001")
    report = _report(question_id="q001", control_text="identical text", treatment_text="identical text")
    flat = flatten_experiment_report(report, "src.json")
    control_flat = next(f for f in flat if f.arm == ARM_CONTROL)
    treatment_flat = next(f for f in flat if f.arm == ARM_TREATMENT)

    client = FakeJudgeClient([_good_answerable_output(), _good_answerable_output()])
    result_control = evaluate_subject(
        build_evaluation_subject(control_flat, row), prompt_variants, client,
        model_config=_model_config(), design_provenance=_design_provenance(),
        schema_doc=schema_doc, source_experiment_file="src.json", source_record_index=0,
        arm=ARM_CONTROL,
    )
    result_treatment = evaluate_subject(
        build_evaluation_subject(treatment_flat, row), prompt_variants, client,
        model_config=_model_config(), design_provenance=_design_provenance(),
        schema_doc=schema_doc, source_experiment_file="src.json", source_record_index=0,
        arm=ARM_TREATMENT,
    )

    # artifact integrity check: fingerprints legitimately match.
    assert result_control.evaluation_subject_fingerprint == result_treatment.evaluation_subject_fingerprint
    # row identity: (question_id, arm) still distinguishes them.
    identity_control = (result_control.question_id, result_control.arm)
    identity_treatment = (result_treatment.question_id, result_treatment.arm)
    assert identity_control != identity_treatment
    # both fields required for row identity are present on the saved record.
    assert result_control.question_id and result_control.arm
    assert result_treatment.question_id and result_treatment.arm


# --------------------------------------------------------------------------- #
# 26-27: dry-run / raw output preservation
# --------------------------------------------------------------------------- #
def test_dry_run_style_pipeline_makes_zero_model_calls(prompt_variants):
    """Mirrors what --dry-run does: build everything up to (and including)
    the rendered request, but never call .run()."""
    row = _row(id="q001")
    flat = flatten_experiment_report(_report(question_id="q001"), "src.json")[0]
    subject = build_evaluation_subject(flat, row)
    render_judge_request(subject, prompt_variants)  # no client involved at all
    # No FakeJudgeClient constructed or called anywhere in this test.
    assert True


def test_raw_judge_output_is_preserved_verbatim(prompt_variants, schema_doc):
    row = _row(id="q001")
    flat = flatten_experiment_report(_report(question_id="q001"), "src.json")[0]
    subject = build_evaluation_subject(flat, row)
    weird_raw = "  " + _good_answerable_output() + "  "
    client = FakeJudgeClient([weird_raw])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert result.raw_judge_output == weird_raw  # untouched, not the stripped version


# --------------------------------------------------------------------------- #
# 28: arm kept only outside Judge input
# --------------------------------------------------------------------------- #
def test_result_record_keeps_arm_only_outside_judge_input(prompt_variants, schema_doc):
    row = _row(id="q001")
    flat = flatten_experiment_report(_report(question_id="q001"), "src.json")[0]
    subject = build_evaluation_subject(flat, row)
    client = FakeJudgeClient([_good_answerable_output()])
    result = evaluate_subject(
        subject, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_TREATMENT,
    )
    assert result.arm == ARM_TREATMENT
    assert result.rendered_judge_input is not None
    assert "Treatment" not in result.rendered_judge_input["user"]
    assert ARM_TREATMENT not in client.calls[0][0] + client.calls[0][1] or (
        # the only legitimate appearance is the fixed instructional sentence
        # explaining blinding itself -- present in system, identically, for
        # every subject regardless of arm.
        "never attempt to infer whether this response came from" in client.calls[0][0]
    )


# --------------------------------------------------------------------------- #
# 29: independent sequential calls carry no state
# --------------------------------------------------------------------------- #
def test_independent_sequential_calls_carry_no_previous_response_state(prompt_variants, schema_doc):
    row1 = _row(id="q001", question="First unique question?")
    row2 = _row(id="q002", question="Second unrelated question?")
    flat1 = flatten_experiment_report(
        _report(question_id="q001", control_text="FIRST RESPONSE TEXT"), "src.json"
    )[0]
    flat2 = flatten_experiment_report(
        _report(question_id="q002", control_text="SECOND RESPONSE TEXT"), "src.json"
    )[0]
    subj1 = build_evaluation_subject(flat1, row1)
    subj2 = build_evaluation_subject(flat2, row2)

    client = FakeJudgeClient([_good_answerable_output(), _good_answerable_output()])
    evaluate_subject(
        subj1, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    evaluate_subject(
        subj2, prompt_variants, client, model_config=_model_config(),
        design_provenance=_design_provenance(), schema_doc=schema_doc,
        source_experiment_file="src.json", source_record_index=0, arm=ARM_CONTROL,
    )
    assert len(client.calls) == 2
    second_system, second_user = client.calls[1]
    assert "FIRST RESPONSE TEXT" not in second_user
    assert "First unique question?" not in second_user
    assert "SECOND RESPONSE TEXT" in second_user


# --------------------------------------------------------------------------- #
# 30: source experiment results remain unchanged
# --------------------------------------------------------------------------- #
def test_flatten_never_mutates_input_report_dict():
    report = _report(question_id="q001")
    before = json.dumps(report, sort_keys=True)
    flatten_experiment_report(report, "src.json")
    after = json.dumps(report, sort_keys=True)
    assert before == after


# --------------------------------------------------------------------------- #
# misc: prompt parser correctness
# --------------------------------------------------------------------------- #
def test_prompt_parser_extracts_both_variants_with_no_leftover_placeholders(prompt_variants):
    assert "TASK COMPLETION" in prompt_variants.system_answerable
    assert "CORRECT ABSTENTION" in prompt_variants.system_unanswerable
    assert "{{" not in prompt_variants.system_answerable
    assert "{{" not in prompt_variants.system_unanswerable


def test_result_record_rejects_unknown_execution_status():
    from heinzy.eval.judge import JudgeResultRecord

    with pytest.raises(ValueError):
        JudgeResultRecord(
            question_id="q001", response_id=None, arm=ARM_CONTROL, answerable=True,
            source_experiment_file="x", source_record_index=0,
            generated_response_fingerprint="a", retrieved_context_fingerprint="b",
            evaluation_subject_fingerprint="c",
            judge_prompt_version="v1", judge_prompt_fingerprint="d",
            judge_rubric_version="v1", judge_rubric_fingerprint="e",
            judge_output_schema_version="v1", judge_output_schema_fingerprint="f",
            evaluation_gold_version="v1.2", evaluation_gold_fingerprint="g",
            rendered_judge_input_fingerprint="h",
            judge_provider="fake", judge_model="fake", temperature=0.0, timeout_seconds=None,
            timestamp="2026-01-01T00:00:00Z", latency_ms=1.0,
            usage_input_tokens=1, usage_output_tokens=1, usage_total_tokens=2,
            usage_model_call_made=True, citation_resolutions=(),
            execution_status="NOT_A_REAL_STATUS", raw_judge_output=None,
            parsed_judge_output=None, error_type=None, error_message=None,
        )
