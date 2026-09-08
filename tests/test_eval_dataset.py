"""Tests for the QA gold-question dataset schema, loader, and validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heinzy.eval.dataset import (
    DatasetError,
    GoldClaim,
    Row,
    SupportingEvidence,
    ValidationError,
    load_dataset,
    policy_unit_split_map,
    validate_dataset,
)


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _raw_answerable_row(**overrides) -> dict:
    row = {
        "id": "ic-04",
        "question": "What courses are required for the AI Management concentration?",
        "answerable": True,
        "question_type": "lookup",
        "question_family_id": "concentration_requirements",
        "policy_unit_id": "mism.concentrations.ai_management",
        "experiment_split": "confirmatory",
        "gold_answer": "36 units: Fundamentals of Operationalizing AI (6), Responsible AI (6), "
                        "Machine Learning for Problem Solving (12), and one AI systems elective (12).",
        "gold_claims": [
            {
                "claim_id": "ic-04-c1",
                "claim": "AI Management requires 36 units of coursework.",
                "supporting_evidence": [
                    {
                        "section_id": "7. Concentrations",
                        "evidence_quote": "A concentration requires completing the designated course set for that area.",
                    },
                    {
                        "section_id": "7.1.1. AI Management Requirements",
                        "evidence_quote": "must complete 36 units of the following coursework",
                    },
                ],
            },
            {
                "claim_id": "ic-04-c2",
                "claim": "Required coursework includes Fundamentals of Operationalizing AI (6 units).",
                "supporting_evidence": [
                    {
                        "section_id": "7.1.1. AI Management Requirements",
                        "evidence_quote": "Fundamentals of Operationalizing AI (6 units)",
                    },
                ],
            },
        ],
        "unanswerable_reason": None,
    }
    row.update(overrides)
    return row


def _raw_unanswerable_row(**overrides) -> dict:
    row = {
        "id": "ooc-02",
        "question": "How many units are required to graduate from the MSCS program?",
        "answerable": False,
        "question_type": "lookup",
        "question_family_id": "program_unit_requirement",
        "policy_unit_id": None,
        "experiment_split": "pilot",
        "gold_answer": None,
        "gold_claims": [],
        "unanswerable_reason": "wrong_scope",
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# loader: valid input
# --------------------------------------------------------------------------- #
def test_load_dataset_valid_answerable_row_with_one_claim(tmp_path):
    row = _raw_answerable_row(gold_claims=[_raw_answerable_row()["gold_claims"][0]])
    path = _write_jsonl(tmp_path, [row])
    rows = load_dataset(path)

    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, Row)
    assert r.answerable is True
    assert r.question_type == "lookup"
    assert r.experiment_split == "confirmatory"
    assert r.unanswerable_reason is None
    assert len(r.gold_claims) == 1
    claim = r.gold_claims[0]
    assert isinstance(claim, GoldClaim)
    assert claim.claim_id == "ic-04-c1"
    assert len(claim.supporting_evidence) == 2


def test_load_dataset_valid_answerable_row_with_multiple_claims(tmp_path):
    path = _write_jsonl(tmp_path, [_raw_answerable_row()])
    rows = load_dataset(path)
    assert len(rows[0].gold_claims) == 2
    assert [c.claim_id for c in rows[0].gold_claims] == ["ic-04-c1", "ic-04-c2"]


def test_load_dataset_claim_with_multiple_supporting_evidence(tmp_path):
    path = _write_jsonl(tmp_path, [_raw_answerable_row()])
    rows = load_dataset(path)
    first_claim = rows[0].gold_claims[0]
    assert len(first_claim.supporting_evidence) == 2
    ev0, ev1 = first_claim.supporting_evidence
    assert isinstance(ev0, SupportingEvidence)
    assert ev0.section_id == "7. Concentrations"
    assert ev0.evidence_quote.startswith("A concentration requires")
    assert ev1.section_id == "7.1.1. AI Management Requirements"


def test_load_dataset_valid_unanswerable_row(tmp_path):
    path = _write_jsonl(tmp_path, [_raw_unanswerable_row()])
    rows = load_dataset(path)
    r = rows[0]
    assert r.answerable is False
    assert r.gold_answer is None
    assert r.gold_claims == []
    assert r.unanswerable_reason == "wrong_scope"
    assert r.policy_unit_id is None


def test_load_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n" + json.dumps(_raw_unanswerable_row()) + "\n\n", encoding="utf-8"
    )
    rows = load_dataset(path)
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# loader: structural / shape errors
# --------------------------------------------------------------------------- #
def test_load_dataset_invalid_json_raises(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DatasetError, match="line 1"):
        load_dataset(path)


def test_load_dataset_missing_required_field_raises(tmp_path):
    row = _raw_answerable_row()
    del row["question_type"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="question_type"):
        load_dataset(path)


def test_load_dataset_row_not_object_raises(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(DatasetError, match="must be a JSON object"):
        load_dataset(path)


def test_load_dataset_wrong_type_raises(tmp_path):
    row = _raw_answerable_row(answerable="true")  # string, not bool
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="answerable"):
        load_dataset(path)


def test_load_dataset_forbidden_gold_evidence_raises(tmp_path):
    row = _raw_answerable_row()
    row["gold_evidence"] = [{"section_id": "7. Concentrations", "evidence_quote": "x"}]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="gold_evidence"):
        load_dataset(path)


@pytest.mark.parametrize(
    "forbidden_field", ["correctness", "faithfulness", "citation_support",
                        "evidence_coverage", "evidence_sufficiency", "hit_at_k",
                        "Hit@K", "mrr", "MRR", "gas", "completeness"],
)
def test_load_dataset_forbidden_derived_metric_raises(tmp_path, forbidden_field):
    row = _raw_answerable_row()
    row[forbidden_field] = 1.0
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError):
        load_dataset(path)


@pytest.mark.parametrize(
    "obsolete_field,value",
    [
        ("primary_category", "multi_section_synthesis"),
        ("tags", ["concentration"]),
        ("reasoning_difficulty", "medium"),
        ("why", "some rationale"),
    ],
)
def test_load_dataset_obsolete_row_fields_rejected(tmp_path, obsolete_field, value):
    row = _raw_answerable_row()
    row[obsolete_field] = value
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match=obsolete_field):
        load_dataset(path)


def test_load_dataset_obsolete_claim_field_essential_rejected(tmp_path):
    row = _raw_answerable_row()
    row["gold_claims"][0]["essential"] = True
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="essential"):
        load_dataset(path)


def test_load_dataset_obsolete_claim_field_supporting_sections_rejected(tmp_path):
    row = _raw_answerable_row()
    row["gold_claims"][0]["supporting_sections"] = []
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="supporting_sections"):
        load_dataset(path)


def test_load_dataset_obsolete_evidence_field_section_path_rejected(tmp_path):
    row = _raw_answerable_row()
    ev = row["gold_claims"][0]["supporting_evidence"][0]
    ev["section_path"] = "7. Concentrations"
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="section_path"):
        load_dataset(path)


def test_load_dataset_obsolete_evidence_field_pages_rejected(tmp_path):
    row = _raw_answerable_row()
    row["gold_claims"][0]["supporting_evidence"][0]["pages"] = [9]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="pages"):
        load_dataset(path)


def test_load_dataset_unexpected_field_raises(tmp_path):
    row = _raw_answerable_row()
    row["totally_unknown_field"] = "oops"
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="unexpected field"):
        load_dataset(path)


def test_load_dataset_claim_missing_field_raises(tmp_path):
    row = _raw_answerable_row()
    del row["gold_claims"][0]["claim"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="claim"):
        load_dataset(path)


def test_load_dataset_evidence_missing_field_raises(tmp_path):
    row = _raw_answerable_row()
    del row["gold_claims"][0]["supporting_evidence"][0]["evidence_quote"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="evidence_quote"):
        load_dataset(path)


# --------------------------------------------------------------------------- #
# validator: builders (bypass the loader to isolate validate_dataset)
# --------------------------------------------------------------------------- #
def _evidence(section_id="7.1.1. AI Management Requirements",
              evidence_quote="must complete 36 units of the following coursework") -> SupportingEvidence:
    return SupportingEvidence(section_id=section_id, evidence_quote=evidence_quote)


def _claim(claim_id="c1", claim="some atomic fact", evidence=None) -> GoldClaim:
    return GoldClaim(
        claim_id=claim_id,
        claim=claim,
        supporting_evidence=evidence if evidence is not None else [_evidence()],
    )


def _answerable(id="ic-04", family="fam-a", split="confirmatory", claims=None, **kw) -> Row:
    return Row(
        id=id,
        question=kw.pop("question", "q?"),
        answerable=True,
        question_type=kw.pop("question_type", "lookup"),
        question_family_id=family,
        policy_unit_id=kw.pop("policy_unit_id", "mism.some.policy"),
        experiment_split=split,
        gold_answer=kw.pop("gold_answer", "the answer"),
        gold_claims=claims if claims is not None else [_claim()],
        unanswerable_reason=kw.pop("unanswerable_reason", None),
        **kw,
    )


def _unanswerable(id="ooc-01", family=None, split="pilot", **kw) -> Row:
    return Row(
        id=id,
        question=kw.pop("question", "q?"),
        answerable=False,
        question_type=kw.pop("question_type", "lookup"),
        question_family_id=family,
        policy_unit_id=None,
        experiment_split=split,
        gold_answer=kw.pop("gold_answer", None),
        gold_claims=kw.pop("gold_claims", []),
        unanswerable_reason=kw.pop("unanswerable_reason", "wrong_scope"),
        **kw,
    )


# --------------------------------------------------------------------------- #
# validator: rules
# --------------------------------------------------------------------------- #
def test_valid_dataset_has_no_errors():
    rows = [_answerable(id="a1", family="fam-a"), _unanswerable(id="o1", family=None)]
    assert validate_dataset(rows) == []


def test_duplicate_id_fails():
    rows = [_answerable(id="dup", family="f1"), _unanswerable(id="dup", family=None)]
    errors = validate_dataset(rows)
    assert any(e.rule == "unique_id" and e.row_id == "dup" for e in errors)


def test_empty_question_fails():
    row = _answerable(id="a1", family="f1", question="   ")
    errors = validate_dataset([row])
    assert any(e.rule == "question_non_empty" for e in errors)


def test_invalid_question_type_fails():
    row = _answerable(id="a1", family="f1", question_type="essay")
    errors = validate_dataset([row])
    assert any(e.rule == "question_type_enum" for e in errors)


@pytest.mark.parametrize("qt", ["lookup", "application", "comparison", "verification", "other"])
def test_all_valid_question_types_pass(qt):
    row = _answerable(id="a1", family="f1", question_type=qt)
    errors = validate_dataset([row])
    assert not any(e.rule == "question_type_enum" for e in errors)


def test_invalid_experiment_split_fails():
    row = _answerable(id="a1", family="f1", split="dev")
    errors = validate_dataset([row])
    assert any(e.rule == "experiment_split_enum" for e in errors)


def test_experiment_split_judge_dev_is_no_longer_valid():
    row = _answerable(id="a1", family="f1", split="judge_dev")
    errors = validate_dataset([row])
    assert any(e.rule == "experiment_split_enum" for e in errors)


def test_answerable_true_missing_gold_answer_fails():
    row = _answerable(id="a1", family="f1", gold_answer=None)
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_gold_answer" for e in errors)


def test_answerable_true_blank_gold_answer_fails():
    row = _answerable(id="a1", family="f1", gold_answer="   ")
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_gold_answer" for e in errors)


def test_answerable_true_zero_gold_claims_fails():
    row = _answerable(id="a1", family="f1", claims=[])
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_gold_claims" for e in errors)


def test_answerable_true_with_unanswerable_reason_fails():
    row = _answerable(id="a1", family="f1", unanswerable_reason="wrong_scope")
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_unanswerable_reason" for e in errors)


def test_answerable_false_with_gold_answer_fails():
    row = _unanswerable(id="o1", gold_answer="should not be here")
    errors = validate_dataset([row])
    assert any(e.rule == "unanswerable_gold_answer" for e in errors)


def test_answerable_false_with_gold_claims_fails():
    row = _unanswerable(id="o1", gold_claims=[_claim()])
    errors = validate_dataset([row])
    assert any(e.rule == "unanswerable_gold_claims" for e in errors)


def test_answerable_false_without_unanswerable_reason_fails():
    row = _unanswerable(id="o1", unanswerable_reason=None)
    errors = validate_dataset([row])
    assert any(e.rule == "unanswerable_reason_required" for e in errors)


def test_invalid_unanswerable_reason_fails():
    row = _unanswerable(id="o1", unanswerable_reason="because reasons")
    errors = validate_dataset([row])
    assert any(e.rule == "unanswerable_reason_required" for e in errors)


@pytest.mark.parametrize(
    "reason", ["wrong_scope", "external_information", "missing_detail", "other"]
)
def test_all_valid_unanswerable_reasons_pass(reason):
    row = _unanswerable(id="o1", unanswerable_reason=reason)
    errors = validate_dataset([row])
    assert not any(e.rule == "unanswerable_reason_required" for e in errors)


def test_duplicate_claim_id_within_row_fails():
    row = _answerable(id="a1", family="f1", claims=[_claim(claim_id="c1"), _claim(claim_id="c1")])
    errors = validate_dataset([row])
    assert any(e.rule == "unique_claim_id" for e in errors)


def test_same_claim_id_in_different_rows_is_fine():
    rows = [
        _answerable(id="a1", family="f1", claims=[_claim(claim_id="shared")]),
        _answerable(id="a2", family="f2", claims=[_claim(claim_id="shared")]),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "unique_claim_id" for e in errors)


def test_empty_claim_string_fails():
    row = _answerable(id="a1", family="f1", claims=[_claim(claim="   ")])
    errors = validate_dataset([row])
    assert any(e.rule == "claim_non_empty" for e in errors)


def test_claim_without_supporting_evidence_fails():
    row = _answerable(id="a1", family="f1", claims=[_claim(evidence=[])])
    errors = validate_dataset([row])
    assert any(e.rule == "claim_supporting_evidence" for e in errors)


def test_empty_section_id_fails():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(evidence=[_evidence(section_id="")])],
    )
    errors = validate_dataset([row])
    assert any(e.rule == "evidence_section_id_non_empty" for e in errors)


def test_whitespace_only_section_id_fails():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(evidence=[_evidence(section_id="   ")])],
    )
    errors = validate_dataset([row])
    assert any(e.rule == "evidence_section_id_non_empty" for e in errors)


def test_empty_evidence_quote_fails():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(evidence=[_evidence(evidence_quote="")])],
    )
    errors = validate_dataset([row])
    assert any(e.rule == "evidence_quote_non_empty" for e in errors)


def test_evidence_quote_present_is_fine():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(evidence=[_evidence(evidence_quote="a real quoted sentence")])],
    )
    errors = validate_dataset([row])
    assert not any(e.rule == "evidence_quote_non_empty" for e in errors)


def test_question_family_id_split_leakage_fails():
    rows = [
        _answerable(id="a1", family="shared-family", split="pilot"),
        _unanswerable(id="o1", family="shared-family", split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    leaks = [e for e in errors if e.rule == "family_split_leakage"]
    assert len(leaks) == 1
    assert "shared-family" in leaks[0].message
    assert leaks[0].row_id is None  # dataset-level, not a single row


def test_question_family_id_same_split_is_fine():
    rows = [
        _answerable(id="a1", family="shared-family", split="confirmatory"),
        _unanswerable(id="o1", family="shared-family", split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "family_split_leakage" for e in errors)


def test_question_family_id_null_is_never_a_leak():
    rows = [
        _unanswerable(id="o1", family=None, split="pilot"),
        _unanswerable(id="o2", family=None, split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "family_split_leakage" for e in errors)


def test_same_policy_unit_id_across_splits_is_allowed():
    rows = [
        _answerable(id="a1", family="f1", split="pilot", policy_unit_id="mism.shared.policy"),
        _answerable(id="a2", family="f2", split="confirmatory", policy_unit_id="mism.shared.policy"),
    ]
    errors = validate_dataset(rows)
    assert errors == []  # rule 11: not a hard constraint, no ValidationError

    overlap = policy_unit_split_map(rows)
    assert overlap["mism.shared.policy"] == {"pilot", "confirmatory"}


def test_policy_unit_split_map_ignores_null_policy_unit_id():
    rows = [_unanswerable(id="o1", family=None, split="pilot")]
    assert policy_unit_split_map(rows) == {}


def test_validation_error_is_immutable_and_stringifies():
    err = ValidationError(row_id="a1", rule="unique_id", message="boom")
    assert "a1" in str(err)
    assert "unique_id" in str(err)
    with pytest.raises(Exception):
        err.rule = "changed"  # frozen dataclass
