"""
Unified evaluation dataset schema, loader, and validator.

pre:  a JSONL file, one question row per line, matching the schema below.
post: load_dataset() returns list[Row] of fully typed objects, or raises
      DatasetError on the first line that does not conform to the schema's
      *shape* (missing/unexpected/forbidden field, wrong JSON type).
      validate_dataset() takes an already-loaded list[Row] and returns every
      ValidationError found (never raises), covering the *semantic* rules
      that only make sense once every row is known: id uniqueness,
      answerable/gold_answer/gold_claims consistency, claim_id uniqueness
      within a row, non-empty content constraints, enum membership, and
      question_family_id/experiment_split leakage.
invariant: this module stores ground truth only. No derived evaluation
           metric (correctness, completeness, faithfulness, citation
           support, evidence coverage/sufficiency, hit@k, MRR, GAS) is a
           field here, and loading a row carrying one of those names fails
           loudly rather than silently dropping it. Row-level gold_evidence
           does not exist either -- evidence provenance lives only inside
           each gold claim's supporting_sections, each of which now also
           carries the evidence_quote it was annotated from.

Two layers, deliberately kept in one small module rather than a generic
schema framework:

  loading    -- can this JSON line even become a Row/GoldClaim/
                SupportingSection object. Structural: required keys present,
                no unexpected or forbidden keys, JSON types match (including
                that tags/pages are homogeneous lists of str/int). Raises
                DatasetError immediately, because a malformed shape can't
                produce a typed object at all.
  validation -- given a list of successfully loaded Rows, do the cross-row
                and content-level semantic rules hold (enum membership,
                non-empty strings, leakage). Returns every violation found
                so a dataset author sees all problems in one pass instead of
                fixing them one crash at a time.

policy_unit_id is explicitly NOT part of validate_dataset()'s error set
(rule 8): it is a balance/leakage *inspection* signal for whoever is
constructing splits, not a hard constraint the loader enforces.

primary_category and tags are annotation/taxonomy fields, independent of
answerable: a false-presupposition question can still be answerable (the
handbook may contain enough evidence to correct the premise), so neither
the loader nor the validator ever infers answerable from them.

reasoning_difficulty is the difficulty of answering GIVEN the necessary
evidence -- it says nothing about retrieval difficulty, which is a separate,
unscored concern for this schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REASONING_DIFFICULTIES = ("easy", "medium", "hard")
EXPERIMENT_SPLITS = ("pilot", "confirmatory")

_ROW_REQUIRED_KEYS = {
    "id", "question", "answerable", "primary_category", "tags",
    "reasoning_difficulty", "policy_unit_id", "question_family_id",
    "gold_answer", "gold_claims", "experiment_split",
}
_ROW_OPTIONAL_KEYS = {"why"}
_ROW_ALLOWED_KEYS = _ROW_REQUIRED_KEYS | _ROW_OPTIONAL_KEYS

_CLAIM_REQUIRED_KEYS = {"claim_id", "claim", "essential", "supporting_sections"}
_SECTION_REQUIRED_KEYS = {"section_path", "pages", "evidence_quote"}

# Named individually so a stale/derived field produces a message that
# explains *why* it's rejected, not just "unexpected field". These are the
# fields rules 9-10 exist to keep out of the dataset.
_FORBIDDEN_ROW_FIELDS = {
    "gold_evidence": "evidence provenance lives only in gold_claims[].supporting_sections, not at row level",
    "correctness": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "completeness": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "faithfulness": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "citation_support": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "evidence_coverage": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "evidence_sufficiency": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "hit_at_k": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "mrr": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "gas": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
}


class DatasetError(ValueError):
    """A JSONL line does not conform to the schema's shape."""


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SupportingSection:
    section_path: str
    pages: list[int]
    evidence_quote: str


@dataclass(frozen=True)
class GoldClaim:
    claim_id: str
    claim: str
    essential: bool
    supporting_sections: list[SupportingSection] = field(default_factory=list)


@dataclass(frozen=True)
class Row:
    id: str
    question: str
    answerable: bool
    primary_category: str
    tags: list[str]
    reasoning_difficulty: str
    policy_unit_id: str | None
    question_family_id: str | None
    gold_answer: str | None
    gold_claims: list[GoldClaim]
    experiment_split: str
    why: str | None = None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _require_type(value: Any, expected_type: type, field_name: str, ctx: str) -> None:
    if not isinstance(value, expected_type):
        raise DatasetError(
            f"{ctx}: field {field_name!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _parse_section(raw: Any, claim_ctx: str, index: int) -> SupportingSection:
    if not isinstance(raw, dict):
        raise DatasetError(
            f"{claim_ctx}: supporting_sections[{index}] must be an object, "
            f"got {type(raw).__name__}"
        )
    missing = _SECTION_REQUIRED_KEYS - raw.keys()
    if missing:
        raise DatasetError(
            f"{claim_ctx}: supporting_sections[{index}] missing required "
            f"field(s): {sorted(missing)}"
        )
    unexpected = raw.keys() - _SECTION_REQUIRED_KEYS
    if unexpected:
        raise DatasetError(
            f"{claim_ctx}: supporting_sections[{index}] unexpected "
            f"field(s): {sorted(unexpected)}"
        )

    section_ctx = f"{claim_ctx} supporting_sections[{index}]"
    _require_type(raw["section_path"], str, "section_path", section_ctx)
    _require_type(raw["pages"], list, "pages", section_ctx)
    for page in raw["pages"]:
        if not isinstance(page, int) or isinstance(page, bool):
            raise DatasetError(f"{section_ctx}: pages must be a list of int, found {page!r}")
    _require_type(raw["evidence_quote"], str, "evidence_quote", section_ctx)

    return SupportingSection(
        section_path=raw["section_path"],
        pages=list(raw["pages"]),
        evidence_quote=raw["evidence_quote"],
    )


def _parse_claim(raw: Any, row_ctx: str, index: int) -> GoldClaim:
    if not isinstance(raw, dict):
        raise DatasetError(
            f"{row_ctx}: gold_claims[{index}] must be an object, got {type(raw).__name__}"
        )
    missing = _CLAIM_REQUIRED_KEYS - raw.keys()
    if missing:
        raise DatasetError(
            f"{row_ctx}: gold_claims[{index}] missing required field(s): {sorted(missing)}"
        )
    unexpected = raw.keys() - _CLAIM_REQUIRED_KEYS
    if unexpected:
        raise DatasetError(
            f"{row_ctx}: gold_claims[{index}] unexpected field(s): {sorted(unexpected)}"
        )

    claim_ctx = f"{row_ctx} gold_claims[{index}]"
    _require_type(raw["claim_id"], str, "claim_id", claim_ctx)
    _require_type(raw["claim"], str, "claim", claim_ctx)
    _require_type(raw["essential"], bool, "essential", claim_ctx)
    _require_type(raw["supporting_sections"], list, "supporting_sections", claim_ctx)

    sections = [
        _parse_section(s, claim_ctx, j) for j, s in enumerate(raw["supporting_sections"])
    ]
    return GoldClaim(
        claim_id=raw["claim_id"],
        claim=raw["claim"],
        essential=raw["essential"],
        supporting_sections=sections,
    )


def _parse_row(raw: Any, line_no: int) -> Row:
    if not isinstance(raw, dict):
        raise DatasetError(f"line {line_no}: row must be a JSON object, got {type(raw).__name__}")

    row_id = raw.get("id")
    ctx = f"line {line_no}" + (f" (id={row_id!r})" if isinstance(row_id, str) else "")

    missing = _ROW_REQUIRED_KEYS - raw.keys()
    if missing:
        raise DatasetError(f"{ctx}: missing required field(s): {sorted(missing)}")

    forbidden_present = [k for k in _FORBIDDEN_ROW_FIELDS if k in raw]
    if forbidden_present:
        details = "; ".join(f"{k!r} ({_FORBIDDEN_ROW_FIELDS[k]})" for k in forbidden_present)
        raise DatasetError(f"{ctx}: forbidden field(s) present: {details}")

    unexpected = raw.keys() - _ROW_ALLOWED_KEYS
    if unexpected:
        raise DatasetError(f"{ctx}: unexpected field(s): {sorted(unexpected)}")

    _require_type(raw["id"], str, "id", ctx)
    _require_type(raw["question"], str, "question", ctx)
    _require_type(raw["answerable"], bool, "answerable", ctx)
    _require_type(raw["primary_category"], str, "primary_category", ctx)
    _require_type(raw["tags"], list, "tags", ctx)
    for tag in raw["tags"]:
        if not isinstance(tag, str):
            raise DatasetError(f"{ctx}: tags must be a list of str, found {tag!r}")
    _require_type(raw["reasoning_difficulty"], str, "reasoning_difficulty", ctx)
    if raw["policy_unit_id"] is not None:
        _require_type(raw["policy_unit_id"], str, "policy_unit_id", ctx)
    if raw["question_family_id"] is not None:
        _require_type(raw["question_family_id"], str, "question_family_id", ctx)
    if raw["gold_answer"] is not None:
        _require_type(raw["gold_answer"], str, "gold_answer", ctx)
    _require_type(raw["gold_claims"], list, "gold_claims", ctx)
    _require_type(raw["experiment_split"], str, "experiment_split", ctx)
    why = raw.get("why")
    if why is not None:
        _require_type(why, str, "why", ctx)

    claims = [_parse_claim(c, ctx, i) for i, c in enumerate(raw["gold_claims"])]

    return Row(
        id=raw["id"],
        question=raw["question"],
        answerable=raw["answerable"],
        primary_category=raw["primary_category"],
        tags=list(raw["tags"]),
        reasoning_difficulty=raw["reasoning_difficulty"],
        policy_unit_id=raw["policy_unit_id"],
        question_family_id=raw["question_family_id"],
        gold_answer=raw["gold_answer"],
        gold_claims=claims,
        experiment_split=raw["experiment_split"],
        why=why,
    )


def load_dataset(path: str | Path) -> list[Row]:
    """Read a JSONL eval dataset into typed Row objects.

    Raises DatasetError on the first line whose JSON shape does not conform
    to the schema (missing/unexpected/forbidden field, wrong JSON type).
    Blank lines are skipped.
    """
    path = Path(path)
    rows: list[Row] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {i}: invalid JSON ({exc})") from exc
        rows.append(_parse_row(raw, i))
    return rows


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationError:
    row_id: str | None
    rule: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display convenience
        where = self.row_id if self.row_id is not None else "<dataset>"
        return f"[{self.rule}] {where}: {self.message}"


def validate_dataset(rows: list[Row]) -> list[ValidationError]:
    """Check the semantic rules that require seeing the whole dataset or a
    row's content (as opposed to its raw JSON shape, which load_dataset
    already enforced).

    Returns every violation found; never raises.
    """
    errors: list[ValidationError] = []

    # Rule 1: id must be unique.
    ids_seen: dict[str, int] = {}
    for r in rows:
        ids_seen[r.id] = ids_seen.get(r.id, 0) + 1
    for id_, count in ids_seen.items():
        if count > 1:
            errors.append(
                ValidationError(id_, "unique_id", f"id {id_!r} is used by {count} rows")
            )

    for r in rows:
        # primary_category must be a non-empty string.
        if not r.primary_category.strip():
            errors.append(
                ValidationError(
                    r.id, "primary_category_non_empty",
                    "primary_category must be a non-empty string",
                )
            )

        # reasoning_difficulty / experiment_split enums.
        if r.reasoning_difficulty not in REASONING_DIFFICULTIES:
            errors.append(
                ValidationError(
                    r.id, "reasoning_difficulty_enum",
                    f"reasoning_difficulty {r.reasoning_difficulty!r} not in "
                    f"{list(REASONING_DIFFICULTIES)}",
                )
            )
        if r.experiment_split not in EXPERIMENT_SPLITS:
            errors.append(
                ValidationError(
                    r.id, "experiment_split_enum",
                    f"experiment_split {r.experiment_split!r} not in {list(EXPERIMENT_SPLITS)}",
                )
            )

        # Rules 2 & 3: answerable <-> gold_answer/gold_claims consistency.
        # Independent of primary_category/tags on purpose -- a
        # false-presupposition (or any other) category never implies
        # answerable one way or the other.
        if r.answerable is False:
            if r.gold_answer is not None:
                errors.append(
                    ValidationError(
                        r.id, "unanswerable_gold_answer",
                        "answerable=false requires gold_answer to be null",
                    )
                )
            if r.gold_claims:
                errors.append(
                    ValidationError(
                        r.id, "unanswerable_gold_claims",
                        "answerable=false requires gold_claims to be empty",
                    )
                )
        else:
            if r.gold_answer is None or not r.gold_answer.strip():
                errors.append(
                    ValidationError(
                        r.id, "answerable_gold_answer",
                        "answerable=true requires a non-empty gold_answer",
                    )
                )
            if not r.gold_claims:
                errors.append(
                    ValidationError(
                        r.id, "answerable_gold_claims",
                        "answerable=true requires at least one gold_claim",
                    )
                )

        # Rule 4: claim_id unique within a row.
        claim_ids = [c.claim_id for c in r.gold_claims]
        dupes = sorted({cid for cid in claim_ids if claim_ids.count(cid) > 1})
        for cid in dupes:
            errors.append(
                ValidationError(
                    r.id, "unique_claim_id",
                    f"claim_id {cid!r} is repeated within this row",
                )
            )

        # Rule 5 + evidence_quote content check, both scoped to answerable
        # rows' claims (unanswerable rows have no gold_claims at all).
        if r.answerable:
            for c in r.gold_claims:
                if not c.supporting_sections:
                    errors.append(
                        ValidationError(
                            r.id, "claim_supporting_sections",
                            f"claim {c.claim_id!r} has no supporting_sections",
                        )
                    )
                for s in c.supporting_sections:
                    if not s.evidence_quote.strip():
                        errors.append(
                            ValidationError(
                                r.id, "claim_evidence_quote",
                                f"claim {c.claim_id!r} has a supporting_section "
                                f"({s.section_path!r}) with an empty evidence_quote",
                            )
                        )

    # Rule 7: rows sharing question_family_id must share an experiment_split.
    family_splits: dict[str, set[str]] = {}
    for r in rows:
        if r.question_family_id is None:
            continue
        family_splits.setdefault(r.question_family_id, set()).add(r.experiment_split)
    for family, splits in family_splits.items():
        if len(splits) > 1:
            errors.append(
                ValidationError(
                    None, "family_split_leakage",
                    f"question_family_id {family!r} spans multiple experiment_splits: "
                    f"{sorted(splits)}",
                )
            )

    return errors


def policy_unit_split_map(rows: list[Row]) -> dict[str, set[str]]:
    """policy_unit_id -> set of experiment_splits it appears in.

    Rule 8: policy_unit_id is NOT a hard split constraint, so this never
    produces a ValidationError. It exists purely so a human constructing or
    reviewing splits can inspect policy-level balance/leakage, e.g. flag a
    policy_unit_id that appears in both pilot and confirmatory.
    """
    out: dict[str, set[str]] = {}
    for r in rows:
        if r.policy_unit_id is None:
            continue
        out.setdefault(r.policy_unit_id, set()).add(r.experiment_split)
    return out
