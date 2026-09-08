"""
QA gold-question dataset: schema, loader, and validator (frozen lean schema).

pre:  a JSONL file, one question row per line, matching the schema below.
post: load_dataset() returns list[Row] of fully typed objects, or raises
      DatasetError on the first line that does not conform to the schema's
      *shape* (missing/unexpected/forbidden field, wrong JSON type).
      validate_dataset() takes an already-loaded list[Row] and returns every
      ValidationError found (never raises), covering the *content* rules
      that only make sense given a value (non-empty strings, enum
      membership) or the whole dataset (id uniqueness,
      question_family_id/experiment_split leakage).
invariant: this dataset stores ground truth only. No derived evaluation
           output (correctness, completeness, faithfulness, citation
           support, evidence coverage/sufficiency, hit@k, MRR, GAS) is a
           field here, and loading a row carrying one of those names fails
           loudly rather than silently dropping it. Row-level gold_evidence
           does not exist either -- evidence provenance lives only inside
           each gold claim's supporting_evidence.
invariant: this is a QA gold-question dataset only. Human response
           annotation and LLM-as-Judge calibration data live in a separate
           dataset (not this module) -- no response-annotation field is
           ever added here.

Schema:

  Row
    id: str                             unique across the dataset
    question: str                       non-empty
    answerable: bool
    question_type: str                  lookup | application | comparison |
                                         verification | other
    question_family_id: str | None      paraphrases of the same target/
                                         entity + same underlying fact/
                                         policy share this id
    policy_unit_id: str | None          broader policy grouping; inspection
                                         only, never a split constraint
    experiment_split: str               pilot | confirmatory
    gold_answer: str | None             non-empty iff answerable
    gold_claims: list[GoldClaim]        non-empty iff answerable
    unanswerable_reason: str | None     required iff not answerable:
                                         wrong_scope | external_information |
                                         missing_detail | other; null iff
                                         answerable

  GoldClaim
    claim_id: str                       unique within its row
    claim: str                          non-empty
    supporting_evidence: list[SupportingEvidence]   non-empty

  SupportingEvidence
    section_id: str                     non-empty
    evidence_quote: str                 non-empty

Two layers, deliberately kept in one small module rather than a generic
schema framework:

  loading    -- can this JSON line even become a Row/GoldClaim/
                SupportingEvidence object. Structural: required keys
                present, no unexpected or forbidden keys, JSON types match.
                Raises DatasetError immediately, because a malformed shape
                can't produce a typed object at all.
  validation -- given a list of successfully loaded Rows, do the content and
                cross-row rules hold. Returns every violation found so a
                dataset author sees all problems in one pass instead of
                fixing them one crash at a time.

question_family_id groups true paraphrases of the same target/entity and the
same underlying fact/policy -- this module only validates that such rows
share an experiment_split (rule 10); it deliberately contains no
semantic-similarity logic to *decide* what counts as a paraphrase, since
that's an authoring decision, not a schema constraint.

policy_unit_id is explicitly NOT part of validate_dataset()'s error set
(rule 11): it is a broader policy-grouping / cluster-aware-analysis signal
for whoever is constructing splits, not a hard constraint the loader
enforces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("lookup", "application", "comparison", "verification", "other")
EXPERIMENT_SPLITS = ("pilot", "confirmatory")
UNANSWERABLE_REASONS = ("wrong_scope", "external_information", "missing_detail", "other")

_ROW_REQUIRED_KEYS = {
    "id", "question", "answerable", "question_type", "question_family_id",
    "policy_unit_id", "experiment_split", "gold_answer", "gold_claims",
    "unanswerable_reason",
}
# No optional row-level fields in the frozen schema (e.g. "why" was removed).
_ROW_ALLOWED_KEYS = _ROW_REQUIRED_KEYS

_CLAIM_REQUIRED_KEYS = {"claim_id", "claim", "supporting_evidence"}
_EVIDENCE_REQUIRED_KEYS = {"section_id", "evidence_quote"}

# Named individually so a stale/obsolete/derived field produces a message
# that explains *why* it's rejected, not just "unexpected field".
_FORBIDDEN_ROW_FIELDS = {
    "gold_evidence": "evidence provenance lives only in gold_claims[].supporting_evidence, not at row level",
    "correctness": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "completeness": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "faithfulness": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "citation_support": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "evidence_coverage": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "evidence_sufficiency": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "hit_at_k": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "Hit@K": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "mrr": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "MRR": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "gas": "derived evaluation metrics are computed by the eval harness and never stored in the dataset",
    "primary_category": "removed from the frozen schema; category/tag annotation fields do not belong in the QA gold dataset",
    "tags": "removed from the frozen schema; category/tag annotation fields do not belong in the QA gold dataset",
    "reasoning_difficulty": "removed from the frozen schema; difficulty is not tracked in the QA gold dataset",
    "why": "removed from the frozen schema; annotation rationale fields do not belong in the QA gold dataset",
}
_FORBIDDEN_CLAIM_FIELDS = {
    "essential": "removed from the frozen schema; GoldClaim no longer carries an essential/optional flag",
    "supporting_sections": "renamed to supporting_evidence",
}
_FORBIDDEN_EVIDENCE_FIELDS = {
    "section_path": "renamed to section_id",
    "pages": "removed from the frozen schema; page numbers are not tracked, use evidence_quote",
}


class DatasetError(ValueError):
    """A JSONL line does not conform to the schema's shape."""


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SupportingEvidence:
    section_id: str
    evidence_quote: str


@dataclass(frozen=True)
class GoldClaim:
    claim_id: str
    claim: str
    supporting_evidence: list[SupportingEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class Row:
    id: str
    question: str
    answerable: bool
    question_type: str
    question_family_id: str | None
    policy_unit_id: str | None
    experiment_split: str
    gold_answer: str | None
    gold_claims: list[GoldClaim]
    unanswerable_reason: str | None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _require_type(value: Any, expected_type: type, field_name: str, ctx: str) -> None:
    if not isinstance(value, expected_type):
        raise DatasetError(
            f"{ctx}: field {field_name!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _parse_evidence(raw: Any, claim_ctx: str, index: int) -> SupportingEvidence:
    if not isinstance(raw, dict):
        raise DatasetError(
            f"{claim_ctx}: supporting_evidence[{index}] must be an object, "
            f"got {type(raw).__name__}"
        )
    missing = _EVIDENCE_REQUIRED_KEYS - raw.keys()
    if missing:
        raise DatasetError(
            f"{claim_ctx}: supporting_evidence[{index}] missing required "
            f"field(s): {sorted(missing)}"
        )
    forbidden_present = [k for k in _FORBIDDEN_EVIDENCE_FIELDS if k in raw]
    if forbidden_present:
        details = "; ".join(
            f"{k!r} ({_FORBIDDEN_EVIDENCE_FIELDS[k]})" for k in forbidden_present
        )
        raise DatasetError(
            f"{claim_ctx}: supporting_evidence[{index}] forbidden field(s) present: {details}"
        )
    unexpected = raw.keys() - _EVIDENCE_REQUIRED_KEYS
    if unexpected:
        raise DatasetError(
            f"{claim_ctx}: supporting_evidence[{index}] unexpected "
            f"field(s): {sorted(unexpected)}"
        )

    evidence_ctx = f"{claim_ctx} supporting_evidence[{index}]"
    _require_type(raw["section_id"], str, "section_id", evidence_ctx)
    _require_type(raw["evidence_quote"], str, "evidence_quote", evidence_ctx)

    return SupportingEvidence(
        section_id=raw["section_id"], evidence_quote=raw["evidence_quote"]
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
    forbidden_present = [k for k in _FORBIDDEN_CLAIM_FIELDS if k in raw]
    if forbidden_present:
        details = "; ".join(
            f"{k!r} ({_FORBIDDEN_CLAIM_FIELDS[k]})" for k in forbidden_present
        )
        raise DatasetError(
            f"{row_ctx}: gold_claims[{index}] forbidden field(s) present: {details}"
        )
    unexpected = raw.keys() - _CLAIM_REQUIRED_KEYS
    if unexpected:
        raise DatasetError(
            f"{row_ctx}: gold_claims[{index}] unexpected field(s): {sorted(unexpected)}"
        )

    claim_ctx = f"{row_ctx} gold_claims[{index}]"
    _require_type(raw["claim_id"], str, "claim_id", claim_ctx)
    _require_type(raw["claim"], str, "claim", claim_ctx)
    _require_type(raw["supporting_evidence"], list, "supporting_evidence", claim_ctx)

    evidence = [
        _parse_evidence(e, claim_ctx, j) for j, e in enumerate(raw["supporting_evidence"])
    ]
    return GoldClaim(claim_id=raw["claim_id"], claim=raw["claim"], supporting_evidence=evidence)


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
    _require_type(raw["question_type"], str, "question_type", ctx)
    if raw["question_family_id"] is not None:
        _require_type(raw["question_family_id"], str, "question_family_id", ctx)
    if raw["policy_unit_id"] is not None:
        _require_type(raw["policy_unit_id"], str, "policy_unit_id", ctx)
    _require_type(raw["experiment_split"], str, "experiment_split", ctx)
    if raw["gold_answer"] is not None:
        _require_type(raw["gold_answer"], str, "gold_answer", ctx)
    _require_type(raw["gold_claims"], list, "gold_claims", ctx)
    if raw["unanswerable_reason"] is not None:
        _require_type(raw["unanswerable_reason"], str, "unanswerable_reason", ctx)

    claims = [_parse_claim(c, ctx, i) for i, c in enumerate(raw["gold_claims"])]

    return Row(
        id=raw["id"],
        question=raw["question"],
        answerable=raw["answerable"],
        question_type=raw["question_type"],
        question_family_id=raw["question_family_id"],
        policy_unit_id=raw["policy_unit_id"],
        experiment_split=raw["experiment_split"],
        gold_answer=raw["gold_answer"],
        gold_claims=claims,
        unanswerable_reason=raw["unanswerable_reason"],
    )


def load_dataset(path: str | Path) -> list[Row]:
    """Read a JSONL QA gold dataset into typed Row objects.

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
    """Check the content and cross-row rules load_dataset() cannot check on
    its own (it only knows a single line's raw JSON shape).

    Returns every violation found; never raises.
    """
    errors: list[ValidationError] = []

    # Rule 1: id must be unique across rows.
    ids_seen: dict[str, int] = {}
    for r in rows:
        ids_seen[r.id] = ids_seen.get(r.id, 0) + 1
    for id_, count in ids_seen.items():
        if count > 1:
            errors.append(
                ValidationError(id_, "unique_id", f"id {id_!r} is used by {count} rows")
            )

    for r in rows:
        # Rule 2: question must be a non-empty string.
        if not r.question.strip():
            errors.append(
                ValidationError(r.id, "question_non_empty", "question must be a non-empty string")
            )

        # Rule 3: question_type enum.
        if r.question_type not in QUESTION_TYPES:
            errors.append(
                ValidationError(
                    r.id, "question_type_enum",
                    f"question_type {r.question_type!r} not in {list(QUESTION_TYPES)}",
                )
            )

        # Rule 4: experiment_split enum.
        if r.experiment_split not in EXPERIMENT_SPLITS:
            errors.append(
                ValidationError(
                    r.id, "experiment_split_enum",
                    f"experiment_split {r.experiment_split!r} not in {list(EXPERIMENT_SPLITS)}",
                )
            )

        # Rules 5 & 6: answerable <-> gold_answer/gold_claims/unanswerable_reason.
        if r.answerable:
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
            if r.unanswerable_reason is not None:
                errors.append(
                    ValidationError(
                        r.id, "answerable_unanswerable_reason",
                        "answerable=true requires unanswerable_reason to be null",
                    )
                )
        else:
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
            if r.unanswerable_reason is None or r.unanswerable_reason not in UNANSWERABLE_REASONS:
                errors.append(
                    ValidationError(
                        r.id, "unanswerable_reason_required",
                        "answerable=false requires unanswerable_reason to be one of "
                        f"{list(UNANSWERABLE_REASONS)}, got {r.unanswerable_reason!r}",
                    )
                )

        # Rule 7: claim_id unique within a row.
        claim_ids = [c.claim_id for c in r.gold_claims]
        dupes = sorted({cid for cid in claim_ids if claim_ids.count(cid) > 1})
        for cid in dupes:
            errors.append(
                ValidationError(
                    r.id, "unique_claim_id",
                    f"claim_id {cid!r} is repeated within this row",
                )
            )

        # Rule 8 (claim non-empty) and rule 9 (supporting_evidence content),
        # plus "every claim needs >=1 supporting_evidence". Checked for
        # whatever claims are present, independent of answerable, so a
        # malformed row surfaces every problem it has, not just the first.
        for c in r.gold_claims:
            if not c.claim.strip():
                errors.append(
                    ValidationError(
                        r.id, "claim_non_empty",
                        f"claim {c.claim_id!r} has an empty claim string",
                    )
                )
            if not c.supporting_evidence:
                errors.append(
                    ValidationError(
                        r.id, "claim_supporting_evidence",
                        f"claim {c.claim_id!r} has no supporting_evidence",
                    )
                )
            for ev in c.supporting_evidence:
                if not ev.section_id.strip():
                    errors.append(
                        ValidationError(
                            r.id, "evidence_section_id_non_empty",
                            f"claim {c.claim_id!r} has a supporting_evidence entry "
                            "with an empty section_id",
                        )
                    )
                if not ev.evidence_quote.strip():
                    errors.append(
                        ValidationError(
                            r.id, "evidence_quote_non_empty",
                            f"claim {c.claim_id!r} has a supporting_evidence entry "
                            f"({ev.section_id!r}) with an empty evidence_quote",
                        )
                    )

    # Rule 10: rows sharing question_family_id must share an experiment_split.
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

    Rule 11: policy_unit_id is NOT a hard split constraint, so this never
    produces a ValidationError. It exists purely so a human constructing or
    reviewing splits can inspect policy-level balance/leakage for broader
    cluster-aware analysis, e.g. flag a policy_unit_id that appears in both
    pilot and confirmatory.
    """
    out: dict[str, set[str]] = {}
    for r in rows:
        if r.policy_unit_id is None:
            continue
        out.setdefault(r.policy_unit_id, set()).add(r.experiment_split)
    return out
