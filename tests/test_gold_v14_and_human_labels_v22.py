"""
Offline validation tests for the q007 human-label correction and q014 gold
correction (v1.3 -> v1.4 gold, v2.1 -> v2.2 human labels). No Judge/LLM/RAG
call is made anywhere in this file -- pure file/dataset/CSV/JSON checks.
"""
from __future__ import annotations

import csv
import json

from heinzy.eval.dataset import load_dataset, validate_dataset

GOLD_V13 = "eval/questions_pilot_v1.3.jsonl"
GOLD_V14 = "eval/questions_pilot_v1.4.jsonl"
LABELS_V21 = "eval/annotation/final_human_labels_v2.1.csv"
LABELS_V22 = "eval/annotation/final_human_labels_v2.2.csv"
METRICS_V21 = "eval/annotation/pilot_30q_v2.1/human_eval_metrics_v2.1.json"
METRICS_V22 = "eval/annotation/pilot_30q_v2.2/human_eval_metrics_v2.2.json"

REQUIRED_AXES = [
    "task_completion", "correctness", "completeness", "faithfulness",
    "citation_validity", "citation_support",
]


def _load_rows_by_id(path: str) -> dict[str, dict]:
    return {json.loads(line)["id"]: json.loads(line) for line in open(path, encoding="utf-8") if line.strip()}


def _load_csv_rows(path: str) -> dict[tuple[str, str], dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["question_id"], r["arm"]): r for r in csv.DictReader(f)}


# --------------------------------------------------------------------------- #
# 1. v1.4 validates 30/30
# --------------------------------------------------------------------------- #
def test_gold_v14_validates_30_of_30():
    rows = load_dataset(GOLD_V14)
    errors = validate_dataset(rows)
    assert len(rows) == 30
    assert errors == []


# --------------------------------------------------------------------------- #
# 2. Exactly q014 differs between v1.3 and v1.4
# --------------------------------------------------------------------------- #
def test_only_q014_differs_between_v13_and_v14():
    v13 = _load_rows_by_id(GOLD_V13)
    v14 = _load_rows_by_id(GOLD_V14)
    assert set(v13) == set(v14) == {f"q{n:03d}" for n in range(1, 31)}
    diffs = [qid for qid in v13 if v13[qid] != v14[qid]]
    assert diffs == ["q014"]


def test_v13_and_v14_metadata_identical_for_q014():
    v13 = _load_rows_by_id(GOLD_V13)["q014"]
    v14 = _load_rows_by_id(GOLD_V14)["q014"]
    for field in ("question", "answerable", "question_type", "question_family_id",
                  "policy_unit_id", "experiment_split", "unanswerable_reason"):
        assert v13[field] == v14[field]
    assert v13["gold_claims"] != v14["gold_claims"]


# --------------------------------------------------------------------------- #
# 3. Exactly q007 Treatment differs between v2.1 and v2.2 human labels
# --------------------------------------------------------------------------- #
def test_only_q007_treatment_differs_between_v21_and_v22_labels():
    v21 = _load_csv_rows(LABELS_V21)
    v22 = _load_csv_rows(LABELS_V22)
    assert set(v21) == set(v22)
    diffs = [key for key in v21 if v21[key] != v22[key]]
    assert diffs == [("q007", "Treatment")]


def test_q007_control_row_unchanged():
    v21 = _load_csv_rows(LABELS_V21)
    v22 = _load_csv_rows(LABELS_V22)
    assert v21[("q007", "Control")] == v22[("q007", "Control")]


# --------------------------------------------------------------------------- #
# 4 & 5. q007 Treatment Completeness = FAIL, Final GAS correctly recomputed
# --------------------------------------------------------------------------- #
def test_q007_treatment_completeness_is_fail_in_v22():
    v22 = _load_csv_rows(LABELS_V22)
    row = v22[("q007", "Treatment")]
    assert row["completeness"] == "FAIL"


def test_q007_treatment_final_gas_recomputed_correctly():
    v21 = _load_csv_rows(LABELS_V21)
    v22 = _load_csv_rows(LABELS_V22)
    old_row = v21[("q007", "Treatment")]
    new_row = v22[("q007", "Treatment")]

    # only completeness (and the derived final_gas) may differ on this row
    for axis in REQUIRED_AXES:
        if axis == "completeness":
            continue
        assert old_row[axis] == new_row[axis], f"{axis} must be unchanged"

    # deterministic GAS recomposition: FAIL iff any required component FAIL
    # (no REVIEW_REQUIRED possible in human labels), else PASS.
    values = [new_row[axis] for axis in REQUIRED_AXES]
    expected_gas = "FAIL" if "FAIL" in values else ("REVIEW_REQUIRED" if "REVIEW_REQUIRED" in values else "PASS")
    assert new_row["final_gas"] == expected_gas == "FAIL"


# --------------------------------------------------------------------------- #
# 6. q014 Control/Treatment human Completeness remains PASS
# --------------------------------------------------------------------------- #
def test_q014_human_completeness_unchanged_pass_both_arms():
    v21 = _load_csv_rows(LABELS_V21)
    v22 = _load_csv_rows(LABELS_V22)
    for arm in ("Control", "Treatment"):
        assert v21[("q014", arm)]["completeness"] == "PASS"
        assert v22[("q014", arm)]["completeness"] == "PASS"
        assert v21[("q014", arm)] == v22[("q014", arm)]


# --------------------------------------------------------------------------- #
# 7. q012 still has exactly q012-c01 and q012-c03 in v1.4
# --------------------------------------------------------------------------- #
def test_q012_two_claim_structure_preserved_in_v14():
    q012 = _load_rows_by_id(GOLD_V14)["q012"]
    claim_ids = [c["claim_id"] for c in q012["gold_claims"]]
    assert claim_ids == ["q012-c01", "q012-c03"]


# --------------------------------------------------------------------------- #
# 8. Historical v1.3 and human v2.1 checksums unchanged (self-consistency:
# re-validating known content rather than an external stored hash, since
# this suite has no separate baseline store -- any accidental in-place edit
# to v1.3/v2.1 would change these known values and fail the test).
# --------------------------------------------------------------------------- #
def test_v13_q014_content_matches_known_pre_correction_wording():
    q014 = _load_rows_by_id(GOLD_V13)["q014"]
    assert q014["gold_claims"][0]["claim"] == (
        "Up to 36 units of relevant graduate-level courses outside Heinz "
        "College may count toward the degree requirements."
    )


def test_v21_q007_treatment_content_matches_known_pre_correction_labels():
    v21 = _load_csv_rows(LABELS_V21)
    row = v21[("q007", "Treatment")]
    assert row["completeness"] == "PASS"
    assert row["final_gas"] == "PASS"


def test_v13_row_count_and_v21_row_count_unchanged():
    assert len(load_dataset(GOLD_V13)) == 30
    with open(LABELS_V21, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 60


# --------------------------------------------------------------------------- #
# 9. Metrics v2.2 are reproducibly derived from labels v2.2
# --------------------------------------------------------------------------- #
def test_metrics_v22_reproducible_from_labels_v22():
    v22 = _load_csv_rows(LABELS_V22)
    metrics = json.load(open(METRICS_V22))

    answerable_qids = sorted(qid for qid, arm in v22 if arm == "Control" and v22[(qid, arm)]["answerable"] == "True")
    unanswerable_qids = sorted(qid for qid, arm in v22 if arm == "Control" and v22[(qid, arm)]["answerable"] == "False")
    assert len(answerable_qids) == 25
    assert len(unanswerable_qids) == 5

    def recompute(qids, arm, field):
        p = sum(1 for qid in qids if v22[(qid, arm)][field] == "PASS")
        f = sum(1 for qid in qids if v22[(qid, arm)][field] == "FAIL")
        na = sum(1 for qid in qids if v22[(qid, arm)][field] == "N/A")
        evaluated_N = p + f
        rate = round(p / evaluated_N, 4) if evaluated_N else None
        return {"PASS": p, "FAIL": f, "N/A": na, "evaluated_N": evaluated_N, "pass_rate_over_evaluated": rate}

    axes = {
        "Task Completion": "task_completion", "Correctness": "correctness",
        "Completeness": "completeness", "Faithfulness": "faithfulness",
        "Citation Validity": "citation_validity", "Citation Support": "citation_support",
        "Final GAS": "final_gas",
    }
    for axis_name, field in axes.items():
        assert recompute(answerable_qids, "Control", field) == metrics["metrics"]["answerable"]["Control"][axis_name]
        assert recompute(answerable_qids, "Treatment", field) == metrics["metrics"]["answerable"]["v2.1_Treatment"][axis_name]

    for axis_name, field in (("Correct Abstention", "correct_abstention"), ("Unsupported Answer", "unsupported_answer")):
        assert recompute(unanswerable_qids, "Control", field) == metrics["metrics"]["unanswerable"]["Control"][axis_name]
        assert recompute(unanswerable_qids, "Treatment", field) == metrics["metrics"]["unanswerable"]["v2.1_Treatment"][axis_name]


def test_metrics_v22_transitions_reproducible():
    v22 = _load_csv_rows(LABELS_V22)
    metrics = json.load(open(METRICS_V22))
    answerable_qids = sorted(qid for qid, arm in v22 if arm == "Control" and v22[(qid, arm)]["answerable"] == "True")

    buckets = {"PASS_to_PASS": [], "FAIL_to_FAIL": [], "FAIL_to_PASS": [], "PASS_to_FAIL": []}
    for qid in answerable_qids:
        c = v22[(qid, "Control")]["final_gas"]
        t = v22[(qid, "Treatment")]["final_gas"]
        buckets[f"{c}_to_{t}"].append(qid)

    assert buckets == metrics["paired_gas_transitions"]["control_to_v21_treatment"]
    assert "q007" in buckets["FAIL_to_FAIL"]
    assert "q019" in buckets["FAIL_to_PASS"]


def test_metrics_v22_provenance_fields():
    metrics = json.load(open(METRICS_V22))
    assert metrics["evaluation_gold_version"] == "v1.4"
    assert metrics["generation_dataset_version"] == "v1.0"  # unchanged historical provenance


def test_metrics_v21_untouched_by_this_task():
    old = json.load(open(METRICS_V21))
    assert old["evaluation_gold_version"] == "v1.2"  # v2.1's own provenance, never rewritten
