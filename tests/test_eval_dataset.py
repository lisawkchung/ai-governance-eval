"""Tests for the unified eval dataset schema, loader, and validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heinzy.eval.dataset import (
    DatasetError,
    GoldClaim,
    Row,
    SupportingSection,
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
        "primary_category": "multi_section_synthesis",
        "tags": ["concentration", "multi_section"],
        "reasoning_difficulty": "medium",
        "policy_unit_id": "mism.concentrations.ai_management",
        "question_family_id": "concentration_requirements",
        "gold_answer": "36 units: Fundamentals of Operationalizing AI (6), Responsible AI (6), "
                        "Machine Learning for Problem Solving (12), and one AI systems elective (12).",
        "gold_claims": [
            {
                "claim_id": "ic-04-c1",
                "claim": "AI Management requires 36 units of coursework.",
                "essential": True,
                "supporting_sections": [
                    {
                        "section_path": "7. Concentrations",
                        "pages": [8],
                        "evidence_quote": "A concentration requires completing the designated course set for that area.",
                    },
                    {
                        "section_path": "7.1.1. AI Management Requirements",
                        "pages": [9],
                        "evidence_quote": "must complete 36 units of the following coursework",
                    },
                ],
            },
            {
                "claim_id": "ic-04-c2",
                "claim": "Required coursework includes Fundamentals of Operationalizing AI (6 units).",
                "essential": True,
                "supporting_sections": [
                    {
                        "section_path": "7.1.1. AI Management Requirements",
                        "pages": [9],
                        "evidence_quote": "Fundamentals of Operationalizing AI (6 units)",
                    },
                ],
            },
        ],
        "experiment_split": "confirmatory",
        "why": "Section 7 states the general rule; 7.1.1 has the AI Management course set.",
    }
    row.update(overrides)
    return row


def _raw_unanswerable_row(**overrides) -> dict:
    row = {
        "id": "ooc-02",
        "question": "How many units are required to graduate from the MSCS program?",
        "answerable": False,
        "primary_category": "adjacent_program",
        "tags": ["trap"],
        "reasoning_difficulty": "hard",
        "policy_unit_id": None,
        "question_family_id": "program_unit_requirement",
        "gold_answer": None,
        "gold_claims": [],
        "experiment_split": "pilot",
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# loader: valid input
# --------------------------------------------------------------------------- #
def test_load_dataset_valid_answerable_and_unanswerable(tmp_path):
    path = _write_jsonl(tmp_path, [_raw_answerable_row(), _raw_unanswerable_row()])
    rows = load_dataset(path)

    assert len(rows) == 2
    answerable, unanswerable = rows

    assert isinstance(answerable, Row)
    assert answerable.answerable is True
    assert answerable.primary_category == "multi_section_synthesis"
    assert answerable.tags == ["concentration", "multi_section"]
    assert answerable.reasoning_difficulty == "medium"
    assert answerable.experiment_split == "confirmatory"
    assert answerable.gold_answer.startswith("36 units")
    assert len(answerable.gold_claims) == 2
    first_claim = answerable.gold_claims[0]
    assert isinstance(first_claim, GoldClaim)
    assert first_claim.claim_id == "ic-04-c1"
    assert len(first_claim.supporting_sections) == 2
    first_section = first_claim.supporting_sections[0]
    assert isinstance(first_section, SupportingSection)
    assert first_section.section_path == "7. Concentrations"
    assert first_section.pages == [8]
    assert first_section.evidence_quote == (
        "A concentration requires completing the designated course set for that area."
    )

    assert unanswerable.answerable is False
    assert unanswerable.primary_category == "adjacent_program"
    assert unanswerable.tags == ["trap"]
    assert unanswerable.experiment_split == "pilot"
    assert unanswerable.gold_answer is None
    assert unanswerable.gold_claims == []
    assert unanswerable.policy_unit_id is None
    assert unanswerable.why is None  # optional field, absent in the raw row


def test_load_dataset_tags_can_be_empty_list(tmp_path):
    path = _write_jsonl(tmp_path, [_raw_unanswerable_row(tags=[])])
    rows = load_dataset(path)
    assert rows[0].tags == []


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
    del row["reasoning_difficulty"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="reasoning_difficulty"):
        load_dataset(path)


def test_load_dataset_old_field_names_are_rejected(tmp_path):
    """category/difficulty/split are the pre-rename names; they must no
    longer be accepted, and simply look like unexpected fields now."""
    row = _raw_answerable_row()
    del row["primary_category"]
    row["category"] = "multi_section_synthesis"
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="missing required field"):
        load_dataset(path)


def test_load_dataset_forbidden_gold_evidence_raises(tmp_path):
    row = _raw_answerable_row()
    row["gold_evidence"] = [{"section_path": "7. Concentrations", "pages": [8]}]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="gold_evidence"):
        load_dataset(path)


@pytest.mark.parametrize(
    "forbidden_field", ["correctness", "faithfulness", "citation_support",
                        "evidence_coverage", "evidence_sufficiency", "hit_at_k",
                        "mrr", "gas", "completeness"],
)
def test_load_dataset_forbidden_derived_metric_raises(tmp_path, forbidden_field):
    row = _raw_answerable_row()
    row[forbidden_field] = 1.0
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match=forbidden_field):
        load_dataset(path)


def test_load_dataset_unexpected_field_raises(tmp_path):
    row = _raw_answerable_row()
    row["totally_unknown_field"] = "oops"
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="unexpected field"):
        load_dataset(path)


def test_load_dataset_wrong_type_raises(tmp_path):
    row = _raw_answerable_row(answerable="true")  # string, not bool
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="answerable"):
        load_dataset(path)


def test_load_dataset_tags_not_a_list_raises(tmp_path):
    row = _raw_answerable_row(tags="concentration")  # string, not a list
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="tags"):
        load_dataset(path)


def test_load_dataset_tags_with_non_string_element_raises(tmp_path):
    row = _raw_answerable_row(tags=["concentration", 5])
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="tags must be a list of str"):
        load_dataset(path)


def test_load_dataset_claim_missing_field_raises(tmp_path):
    row = _raw_answerable_row()
    del row["gold_claims"][0]["essential"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="essential"):
        load_dataset(path)


def test_load_dataset_claim_unexpected_field_raises(tmp_path):
    row = _raw_answerable_row()
    row["gold_claims"][0]["confidence"] = 0.9
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="unexpected field"):
        load_dataset(path)


def test_load_dataset_section_missing_field_raises(tmp_path):
    row = _raw_answerable_row()
    del row["gold_claims"][0]["supporting_sections"][0]["pages"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="pages"):
        load_dataset(path)


def test_load_dataset_section_missing_evidence_quote_raises(tmp_path):
    row = _raw_answerable_row()
    del row["gold_claims"][0]["supporting_sections"][0]["evidence_quote"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="evidence_quote"):
        load_dataset(path)


def test_load_dataset_section_evidence_quote_wrong_type_raises(tmp_path):
    row = _raw_answerable_row()
    row["gold_claims"][0]["supporting_sections"][0]["evidence_quote"] = 123
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="evidence_quote"):
        load_dataset(path)


def test_load_dataset_section_bad_page_type_raises(tmp_path):
    row = _raw_answerable_row()
    row["gold_claims"][0]["supporting_sections"][0]["pages"] = ["eight"]
    path = _write_jsonl(tmp_path, [row])
    with pytest.raises(DatasetError, match="pages must be a list of int"):
        load_dataset(path)


def test_load_dataset_row_not_object_raises(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(DatasetError, match="must be a JSON object"):
        load_dataset(path)


# --------------------------------------------------------------------------- #
# validator: builders (bypass the loader to isolate validate_dataset)
# --------------------------------------------------------------------------- #
def _section(
    section_path="7.1.1. AI Management Requirements",
    pages=(9,),
    evidence_quote="must complete 36 units of the following coursework",
) -> SupportingSection:
    return SupportingSection(
        section_path=section_path, pages=list(pages), evidence_quote=evidence_quote
    )


def _claim(claim_id="c1", essential=True, sections=None) -> GoldClaim:
    return GoldClaim(
        claim_id=claim_id,
        claim="some atomic fact",
        essential=essential,
        supporting_sections=sections if sections is not None else [_section()],
    )


def _answerable(id="ic-04", family="fam-a", split="confirmatory", claims=None, **kw) -> Row:
    return Row(
        id=id,
        question="q?",
        answerable=True,
        primary_category=kw.pop("primary_category", "direct_lookup"),
        tags=kw.pop("tags", []),
        reasoning_difficulty=kw.pop("reasoning_difficulty", "medium"),
        policy_unit_id=kw.pop("policy_unit_id", "mism.some.policy"),
        question_family_id=family,
        gold_answer=kw.pop("gold_answer", "the answer"),
        gold_claims=claims if claims is not None else [_claim()],
        experiment_split=split,
        **kw,
    )


def _unanswerable(id="ooc-01", family=None, split="pilot", **kw) -> Row:
    return Row(
        id=id,
        question="q?",
        answerable=False,
        primary_category=kw.pop("primary_category", "world_knowledge"),
        tags=kw.pop("tags", []),
        reasoning_difficulty=kw.pop("reasoning_difficulty", "easy"),
        policy_unit_id=None,
        question_family_id=family,
        gold_answer=kw.pop("gold_answer", None),
        gold_claims=kw.pop("gold_claims", []),
        experiment_split=split,
        **kw,
    )


# --------------------------------------------------------------------------- #
# validator: rules
# --------------------------------------------------------------------------- #
def test_valid_dataset_has_no_errors():
    rows = [_answerable(id="a1", family="fam-a"), _unanswerable(id="o1", family=None)]
    assert validate_dataset(rows) == []


def test_rule1_duplicate_id_detected():
    rows = [_answerable(id="dup", family="f1"), _unanswerable(id="dup", family=None)]
    errors = validate_dataset(rows)
    assert any(e.rule == "unique_id" and e.row_id == "dup" for e in errors)


def test_rule2_unanswerable_with_gold_answer_flagged():
    row = _unanswerable(id="o1", gold_answer="should not be here")
    errors = validate_dataset([row])
    assert any(e.rule == "unanswerable_gold_answer" for e in errors)


def test_rule2_unanswerable_with_gold_claims_flagged():
    row = _unanswerable(id="o1", gold_claims=[_claim()])
    errors = validate_dataset([row])
    assert any(e.rule == "unanswerable_gold_claims" for e in errors)


def test_rule3_answerable_missing_gold_answer_flagged():
    row = _answerable(id="a1", family="f1", gold_answer=None)
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_gold_answer" for e in errors)


def test_rule3_answerable_blank_gold_answer_flagged():
    row = _answerable(id="a1", family="f1", gold_answer="   ")
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_gold_answer" for e in errors)


def test_rule3_answerable_empty_gold_claims_flagged():
    row = _answerable(id="a1", family="f1", claims=[])
    errors = validate_dataset([row])
    assert any(e.rule == "answerable_gold_claims" for e in errors)


def test_rule4_duplicate_claim_id_within_row_flagged():
    row = _answerable(id="a1", family="f1", claims=[_claim(claim_id="c1"), _claim(claim_id="c1")])
    errors = validate_dataset([row])
    assert any(e.rule == "unique_claim_id" for e in errors)


def test_rule4_same_claim_id_in_different_rows_is_fine():
    rows = [
        _answerable(id="a1", family="f1", claims=[_claim(claim_id="shared")]),
        _answerable(id="a2", family="f2", claims=[_claim(claim_id="shared")]),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "unique_claim_id" for e in errors)


def test_rule5_claim_without_supporting_sections_flagged_when_answerable():
    row = _answerable(id="a1", family="f1", claims=[_claim(sections=[])])
    errors = validate_dataset([row])
    assert any(e.rule == "claim_supporting_sections" for e in errors)


def test_primary_category_empty_string_flagged():
    row = _answerable(id="a1", family="f1", primary_category="")
    errors = validate_dataset([row])
    assert any(e.rule == "primary_category_non_empty" for e in errors)


def test_primary_category_whitespace_only_flagged():
    row = _unanswerable(id="o1", primary_category="   ")
    errors = validate_dataset([row])
    assert any(e.rule == "primary_category_non_empty" for e in errors)


def test_evidence_quote_empty_flagged_for_answerable_claim():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(sections=[_section(evidence_quote="")])],
    )
    errors = validate_dataset([row])
    assert any(e.rule == "claim_evidence_quote" for e in errors)


def test_evidence_quote_whitespace_only_flagged():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(sections=[_section(evidence_quote="   ")])],
    )
    errors = validate_dataset([row])
    assert any(e.rule == "claim_evidence_quote" for e in errors)


def test_evidence_quote_present_is_fine():
    row = _answerable(
        id="a1", family="f1",
        claims=[_claim(sections=[_section(evidence_quote="a real quoted sentence")])],
    )
    errors = validate_dataset([row])
    assert not any(e.rule == "claim_evidence_quote" for e in errors)


def test_reasoning_difficulty_invalid_flagged():
    row = _answerable(id="a1", family="f1", reasoning_difficulty="impossible")
    errors = validate_dataset([row])
    assert any(e.rule == "reasoning_difficulty_enum" for e in errors)


def test_experiment_split_invalid_flagged():
    row = _answerable(id="a1", family="f1", split="dev")
    errors = validate_dataset([row])
    assert any(e.rule == "experiment_split_enum" for e in errors)


def test_experiment_split_judge_dev_is_no_longer_valid():
    """judge_dev/judge_validation_holdout belong to a separate judge
    calibration dataset now, not this question schema."""
    row = _answerable(id="a1", family="f1", split="judge_dev")
    errors = validate_dataset([row])
    assert any(e.rule == "experiment_split_enum" for e in errors)


def test_experiment_split_pilot_and_confirmatory_are_valid():
    rows = [
        _answerable(id="a1", family="f1", split="pilot"),
        _answerable(id="a2", family="f2", split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "experiment_split_enum" for e in errors)


def test_family_split_leakage_detected():
    rows = [
        _answerable(id="a1", family="shared-family", split="pilot"),
        _unanswerable(id="o1", family="shared-family", split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    leaks = [e for e in errors if e.rule == "family_split_leakage"]
    assert len(leaks) == 1
    assert "shared-family" in leaks[0].message
    assert leaks[0].row_id is None  # dataset-level, not a single row


def test_family_same_split_is_fine():
    rows = [
        _answerable(id="a1", family="shared-family", split="confirmatory"),
        _unanswerable(id="o1", family="shared-family", split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "family_split_leakage" for e in errors)


def test_family_null_family_id_is_never_a_leak():
    rows = [
        _unanswerable(id="o1", family=None, split="pilot"),
        _unanswerable(id="o2", family=None, split="confirmatory"),
    ]
    errors = validate_dataset(rows)
    assert not any(e.rule == "family_split_leakage" for e in errors)


def test_policy_unit_id_across_splits_is_not_a_validation_error():
    rows = [
        _answerable(id="a1", family="f1", split="pilot", policy_unit_id="mism.shared.policy"),
        _answerable(id="a2", family="f2", split="confirmatory", policy_unit_id="mism.shared.policy"),
    ]
    errors = validate_dataset(rows)
    assert errors == []  # rule 8: not a hard constraint, no ValidationError

    overlap = policy_unit_split_map(rows)
    assert overlap["mism.shared.policy"] == {"pilot", "confirmatory"}


def test_policy_unit_split_map_ignores_null_policy_unit_id():
    rows = [_unanswerable(id="o1", family=None, split="pilot")]
    assert policy_unit_split_map(rows) == {}


def test_false_presupposition_category_does_not_imply_answerable_false():
    """answerable is independent of primary_category/tags: a
    false-presupposition question can still be answerable if the handbook
    has enough evidence to correct the false premise."""
    row = _answerable(
        id="a1", family="f1",
        primary_category="false_presupposition",
        tags=["false_presupposition", "correction"],
    )
    assert row.answerable is True
    errors = validate_dataset([row])
    assert errors == []


def test_false_presupposition_category_can_also_be_unanswerable():
    """The reverse also holds: the category alone doesn't force either
    answerability outcome."""
    row = _unanswerable(
        id="o1",
        primary_category="false_presupposition",
        tags=["false_presupposition"],
    )
    assert row.answerable is False
    errors = validate_dataset([row])
    assert errors == []


def test_validation_error_is_immutable_and_stringifies():
    err = ValidationError(row_id="a1", rule="unique_id", message="boom")
    assert "a1" in str(err)
    assert "unique_id" in str(err)
    with pytest.raises(Exception):
        err.rule = "changed"  # frozen dataclass
