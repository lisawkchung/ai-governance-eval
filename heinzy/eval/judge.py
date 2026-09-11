"""
Judge v1 execution module (Heinzy RAG evaluation).

MODULE BOUNDARY -- READ BEFORE ADDING ANYTHING TO THIS FILE:

  Judge execution (THIS MODULE):
      saved experiment response + gold v1.2 + retrieved evidence
      -> deterministic citation resolution -> blinded Judge input
      -> Judge model call -> parsed/validated Judge result

  Calibration comparison (A SEPARATE, LATER MODULE/SCRIPT -- not this one):
      Judge result + independently finalized human label
      -> agreement/error analysis

This module NEVER imports, opens, or reads
`eval/annotation/final_human_labels_v2.1.csv` (or any other human-label /
human-annotation file), and never accepts one as a parameter. Grep this file
for "final_human_labels" or "human_reviewed" if you are ever unsure --
neither string should appear anywhere below except in this docstring.
Comparing Judge output to human labels is a distinct, later step performed
by a different module/script. That later join has TWO SEPARATE concepts,
never to be merged into one:

  ROW IDENTITY      -- (question_id, arm). This is what identifies WHICH
                        Control or Treatment evaluation row a human label
                        and a Judge result both refer to. The later join
                        must match rows on this pair.

  ARTIFACT INTEGRITY -- `evaluation_subject_fingerprint` (see below). This
                        verifies that the Judge evaluated the SAME
                        generated_response/retrieved_context/citation-
                        resolution artifact the human label was written
                        against. It is a validation check performed AFTER
                        the (question_id, arm) join, never the join key
                        itself: Control and Treatment can legitimately
                        produce identical generated text, retrieved
                        context, and citation mapping (e.g. an unanswered
                        zero-hit question), in which case their
                        `evaluation_subject_fingerprint` values are equal
                        by construction, and a fingerprint-only join would
                        silently conflate two distinct rows. The later
                        module must: (1) join by (question_id, arm), (2)
                        compare the resulting Judge result's
                        `evaluation_subject_fingerprint` against the
                        fingerprint the human label was recorded against,
                        and (3) fail or flag the row if that fingerprint
                        does not match -- never treat a fingerprint match
                        alone as sufficient to identify a row.

BLINDING INVARIANT: the actual content sent to the Judge model (built by
`render_judge_request`, from an `EvaluationSubject`) structurally cannot
carry Control/Treatment arm identity, treatment-strategy/verifier metadata,
zero-hit-guard metadata, SUT model identity, human labels, or any prior
Judge output -- `EvaluationSubject` simply has no field for any of those, so
there is no code path through which one could leak into a rendered prompt.
Arm/response identity lives only in orchestration metadata
(`FlattenedResponse`, `JudgeResultRecord`), which is never rendered into a
prompt.

INDEPENDENCE INVARIANT: every `evaluate_subject()` call is fully
self-contained -- one system message, one user message, built fresh from one
`EvaluationSubject`. No conversation history, prior response, prior Judge
rationale, or running summary is ever threaded between calls.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import jsonschema

from heinzy.eval.citation_resolution import (
    AMBIGUOUS,
    RESOLVED,
    CitationResolution,
    RetrievedSectionChunk,
    resolve_citations,
)
from heinzy.eval.dataset import GoldClaim, Row
from heinzy.eval.experiment import RetrievalChunkSnapshot
from heinzy.generation.generator import extract_usage
from heinzy.generation.grounding import extract_citations

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
ARM_CONTROL = "Control"
ARM_TREATMENT = "Treatment"
DEFAULT_ARMS = (ARM_CONTROL, ARM_TREATMENT)

JUDGE_PROMPT_VERSION = "v1"
JUDGE_RUBRIC_VERSION = "v1"
JUDGE_OUTPUT_SCHEMA_VERSION = "v1"

EXECUTION_SUCCESS = "SUCCESS"
EXECUTION_ERROR = "EXECUTION_ERROR"
PARSE_ERROR = "PARSE_ERROR"
SCHEMA_ERROR = "SCHEMA_ERROR"
_VALID_EXECUTION_STATUSES = (EXECUTION_SUCCESS, EXECUTION_ERROR, PARSE_ERROR, SCHEMA_ERROR)


class JudgeInputError(ValueError):
    """Raised on any of the loud-failure conditions in the task spec
    (duplicate ids, missing arm coverage, gold-join failure, Control-source
    mismatch, etc). `.errors` always carries every problem found, not just
    the first."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# --------------------------------------------------------------------------- #
# fingerprint helpers
# --------------------------------------------------------------------------- #
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_text(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# flattening saved experiment results into per-arm response records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlattenedResponse:
    question_id: str
    arm: str
    generated_response: str
    retrieved_chunks: tuple[RetrievalChunkSnapshot, ...]
    source_experiment_file: str
    source_record_index: int
    # Not present in the saved experiment-result schema (see final report) --
    # kept as an explicit field, always None today, so a future schema that
    # DOES carry a stable response_id needs no shape change here.
    response_id: str | None = None


def _chunk_from_dict(c: dict[str, Any]) -> RetrievalChunkSnapshot:
    return RetrievalChunkSnapshot(
        rank=c["rank"],
        chunk_id=c["chunk_id"],
        score=c["score"],
        doc_id=c["doc_id"],
        section_path=c.get("section_path"),
        source_pages=list(c.get("source_pages") or []),
        text=c["text"],
    )


def flatten_experiment_report(
    report: dict[str, Any],
    source_experiment_file: str,
    *,
    arms: tuple[str, ...] = DEFAULT_ARMS,
) -> list[FlattenedResponse]:
    """One FlattenedResponse per (result, requested arm). Duplicate
    question_id across Control and Treatment is expected and NOT an error --
    that is exactly the paired-arm design; uniqueness is enforced on
    (question_id, arm), not question_id alone (see assert_unique_flattened).
    """
    out: list[FlattenedResponse] = []
    for idx, r in enumerate(report.get("results", [])):
        qid = r["question_id"]
        chunks = tuple(_chunk_from_dict(c) for c in r["retrieval"]["chunks"])
        if ARM_CONTROL in arms:
            out.append(
                FlattenedResponse(
                    question_id=qid,
                    arm=ARM_CONTROL,
                    generated_response=r["control"]["text"],
                    retrieved_chunks=chunks,
                    source_experiment_file=source_experiment_file,
                    source_record_index=idx,
                )
            )
        if ARM_TREATMENT in arms:
            out.append(
                FlattenedResponse(
                    question_id=qid,
                    arm=ARM_TREATMENT,
                    generated_response=r["treatment"]["final_text"],
                    retrieved_chunks=chunks,
                    source_experiment_file=source_experiment_file,
                    source_record_index=idx,
                )
            )
    return out


def assert_unique_flattened(flattened: list[FlattenedResponse]) -> None:
    """Rule B: (question_id, arm) must be unique; response_id must be
    unique when present. Duplicate question_id ACROSS arms is allowed and
    checked nowhere here."""
    counts: dict[tuple[str, str], int] = {}
    for f in flattened:
        key = (f.question_id, f.arm)
        counts[key] = counts.get(key, 0) + 1
    problems = [
        f"duplicate evaluation subject: question_id={qid!r} arm={arm!r} appears {n} times"
        for (qid, arm), n in counts.items()
        if n > 1
    ]

    response_ids = [f.response_id for f in flattened if f.response_id is not None]
    for rid in sorted({r for r in response_ids if response_ids.count(r) > 1}):
        problems.append(f"duplicate response_id={rid!r}")

    if problems:
        raise JudgeInputError(problems)


def validate_full_coverage(
    flattened: list[FlattenedResponse],
    gold_rows: list[Row],
    *,
    expected_arms: tuple[str, ...] = DEFAULT_ARMS,
) -> None:
    """Rule C: for a full run, every gold question_id must have exactly one
    response per expected arm, and every response must join to a gold row."""
    gold_ids = {r.id for r in gold_rows}
    present: dict[str, set[str]] = {}
    extra_ids: set[str] = set()
    for f in flattened:
        if f.question_id not in gold_ids:
            extra_ids.add(f.question_id)
            continue
        present.setdefault(f.question_id, set()).add(f.arm)

    problems = [
        f"response for question_id={qid!r} has no matching gold row"
        for qid in sorted(extra_ids)
    ]
    for qid in sorted(gold_ids):
        missing = set(expected_arms) - present.get(qid, set())
        if missing:
            problems.append(f"question_id={qid!r} missing arm(s): {sorted(missing)}")
    if problems:
        raise JudgeInputError(problems)


# --------------------------------------------------------------------------- #
# gold join
# --------------------------------------------------------------------------- #
def build_gold_index(rows: list[Row]) -> dict[str, Row]:
    """Rule A: gold question_id must be unique. Independent of
    heinzy.eval.dataset.validate_dataset's own uniqueness rule, so this
    module never silently trusts an unvalidated rows list."""
    index: dict[str, Row] = {}
    dupes: list[str] = []
    for r in rows:
        if r.id in index:
            dupes.append(r.id)
        index[r.id] = r
    if dupes:
        raise JudgeInputError(
            [f"duplicate gold question_id: {qid!r}" for qid in sorted(set(dupes))]
        )
    return index


def join_gold(flat: FlattenedResponse, gold_index: dict[str, Row]) -> Row:
    row = gold_index.get(flat.question_id)
    if row is None:
        raise JudgeInputError(
            [f"response question_id={flat.question_id!r} arm={flat.arm!r} "
             "has no matching gold row"]
        )
    return row


# --------------------------------------------------------------------------- #
# Control-source consistency check (v1 report vs v2.1 report)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ControlComparisonResult:
    question_id: str
    text_match: bool
    context_match: bool
    mismatch_categories: tuple[str, ...] = field(default_factory=tuple)


def _control_context_repr(chunks: list[dict[str, Any]]) -> str:
    return "\n---\n".join(
        f"{c.get('section_path')}|{c.get('doc_id')}|{c.get('text')}" for c in chunks
    )


def compare_control_sources(
    report_a: dict[str, Any], report_b: dict[str, Any]
) -> list[ControlComparisonResult]:
    """Question-by-question comparison of Control generated text AND
    retrieved context between two saved experiment reports. Pure comparison
    -- makes no decision about which source to prefer."""
    by_id_a = {r["question_id"]: r for r in report_a.get("results", [])}
    by_id_b = {r["question_id"]: r for r in report_b.get("results", [])}
    out: list[ControlComparisonResult] = []
    for qid in sorted(set(by_id_a) | set(by_id_b)):
        a, b = by_id_a.get(qid), by_id_b.get(qid)
        if a is None or b is None:
            out.append(ControlComparisonResult(qid, False, False, ("MISSING_IN_ONE_SOURCE",)))
            continue
        text_match = a["control"]["text"] == b["control"]["text"]
        context_match = (
            _control_context_repr(a["retrieval"]["chunks"])
            == _control_context_repr(b["retrieval"]["chunks"])
        )
        categories: list[str] = []
        if not text_match:
            categories.append("CONTROL_TEXT_DIFFERS")
        if not context_match:
            categories.append("RETRIEVED_CONTEXT_DIFFERS")
        out.append(ControlComparisonResult(qid, text_match, context_match, tuple(categories)))
    return out


def assert_control_sources_consistent(comparisons: list[ControlComparisonResult]) -> None:
    """CRITICAL per task section 5: never silently choose/merge on mismatch.
    Raises listing every mismatched question_id and its exact category."""
    problems = [
        f"question_id={c.question_id!r}: {list(c.mismatch_categories)}"
        for c in comparisons
        if c.mismatch_categories
    ]
    if problems:
        raise JudgeInputError(problems)


# --------------------------------------------------------------------------- #
# citation extraction + resolution glue
# --------------------------------------------------------------------------- #
def extract_raw_citations(text: str) -> list[str]:
    """Candidate-citation-string EXTRACTION only, reusing the same
    "(see ...)"/prose-section regex parsing the runtime pipeline already
    uses (heinzy.generation.grounding.extract_citations), so Judge input
    reflects the same citation strings the system itself treats as
    citations. This function makes NO validity/resolution judgment -- that
    is heinzy.eval.citation_resolution.resolve_citations' job exclusively
    (exact-match-only; grounding.is_supported/unsupported_citations, which
    does loose bidirectional substring matching, is deliberately NOT reused
    for resolution -- see citation_resolution.py's module docstring).
    """
    return extract_citations(text)


def build_citation_resolutions(
    generated_response: str, retrieved_chunks: tuple[RetrievalChunkSnapshot, ...]
) -> list[CitationResolution]:
    raw_citations = extract_raw_citations(generated_response)
    resolution_chunks = [
        RetrievedSectionChunk(
            section_path=c.section_path, doc_id=c.doc_id, text=c.text, rank=c.rank
        )
        for c in retrieved_chunks
    ]
    return resolve_citations(raw_citations, resolution_chunks)


# --------------------------------------------------------------------------- #
# evaluation subject (the blinded content unit)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluationSubject:
    """Exactly the fields the Judge is allowed to see. No arm, no verifier
    metadata, no human label, no prior Judge output -- there is no field for
    any of those, by construction."""

    question_id: str
    question: str
    answerable: bool
    generated_response: str
    retrieved_chunks: tuple[RetrievalChunkSnapshot, ...]
    citation_resolutions: tuple[CitationResolution, ...]
    gold_answer: str | None
    gold_claims: tuple[GoldClaim, ...] | None
    unanswerable_reason: str | None


def build_evaluation_subject(flat: FlattenedResponse, gold_row: Row) -> EvaluationSubject:
    if gold_row.id != flat.question_id:
        raise JudgeInputError(
            [f"gold row id {gold_row.id!r} does not match response "
             f"question_id {flat.question_id!r}"]
        )
    if not isinstance(gold_row.answerable, bool):
        raise JudgeInputError(
            [f"question_id={flat.question_id!r}: gold answerable flag missing/invalid"]
        )

    citation_resolutions = tuple(
        build_citation_resolutions(flat.generated_response, flat.retrieved_chunks)
    )

    if gold_row.answerable:
        if not gold_row.gold_claims:
            raise JudgeInputError(
                [f"question_id={flat.question_id!r}: answerable=true but gold_claims is empty"]
            )
        return EvaluationSubject(
            question_id=flat.question_id,
            question=gold_row.question,
            answerable=True,
            generated_response=flat.generated_response,
            retrieved_chunks=flat.retrieved_chunks,
            citation_resolutions=citation_resolutions,
            gold_answer=gold_row.gold_answer,
            gold_claims=tuple(gold_row.gold_claims),
            unanswerable_reason=None,
        )

    return EvaluationSubject(
        question_id=flat.question_id,
        question=gold_row.question,
        answerable=False,
        generated_response=flat.generated_response,
        retrieved_chunks=flat.retrieved_chunks,
        citation_resolutions=citation_resolutions,
        gold_answer=None,
        gold_claims=None,
        unanswerable_reason=gold_row.unanswerable_reason,
    )


def _chunk_to_dict(c: RetrievalChunkSnapshot) -> dict[str, Any]:
    return {
        "rank": c.rank, "chunk_id": c.chunk_id, "score": c.score, "doc_id": c.doc_id,
        "section_path": c.section_path, "source_pages": list(c.source_pages), "text": c.text,
    }


def _citation_resolution_to_dict(r: CitationResolution) -> dict[str, Any]:
    return {
        "raw_citation": r.raw_citation,
        "normalized_citation": r.normalized_citation,
        "resolution_status": r.resolution_status,
        "resolved_section_id": r.resolved_section_id,
        "resolved_section_path": r.resolved_section_path,
        "resolved_doc_id": r.resolved_doc_id,
        "resolved_retrieved_text": r.resolved_retrieved_text,
        # Hierarchy-aware allowed evidence scope (cited section + every
        # retrieved descendant, per the adopted citation-scope policy --
        # see heinzy.eval.citation_resolution module docstring). Citation
        # Support must be judged against THIS scope, not resolved_retrieved_text
        # alone, which is only the cited node's own text.
        "resolved_scope_section_ids": list(r.resolved_scope_section_ids),
        "resolved_scope_retrieved_text": r.resolved_scope_retrieved_text,
        "ambiguous_candidates": list(r.ambiguous_candidates),
    }


def fingerprint_retrieved_context(chunks: tuple[RetrievalChunkSnapshot, ...]) -> str:
    payload = [_chunk_to_dict(c) for c in chunks]
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def evaluation_subject_fingerprint(subject: EvaluationSubject) -> str:
    """Binds question_id + generated_response + retrieved_context + resolved
    citation mapping. Deliberately excludes arm -- two subjects built from
    the same underlying response/context/citations fingerprint identically
    regardless of which arm produced them.

    NOT a unique row-identity key: because it excludes arm, Control and
    Treatment CAN legitimately share the same fingerprint (e.g. an identical
    response/context on a zero-hit question). Row identity for a later
    human-label join is `(question_id, arm)`; this fingerprint is an
    artifact-integrity check applied AFTER that join, never a substitute
    for it. See the module docstring's ROW IDENTITY / ARTIFACT INTEGRITY
    section.
    """
    payload = {
        "question_id": subject.question_id,
        "generated_response": subject.generated_response,
        "retrieved_context": [_chunk_to_dict(c) for c in subject.retrieved_chunks],
        "citation_resolutions": [
            _citation_resolution_to_dict(r) for r in subject.citation_resolutions
        ],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# rendering: gold / retrieved-context / citations -> plain text blocks
# --------------------------------------------------------------------------- #
def render_gold_claims(claims: tuple[GoldClaim, ...] | list[GoldClaim]) -> str:
    if not claims:
        return "(none)"
    return "\n".join(f"{i}. [{c.claim_id}] {c.claim}" for i, c in enumerate(claims, start=1))


def render_supporting_evidence(claims: tuple[GoldClaim, ...] | list[GoldClaim]) -> str:
    if not claims:
        return "(none)"
    lines = [
        f'- [{c.claim_id}] section {ev.section_id!r}: "{ev.evidence_quote}"'
        for c in claims
        for ev in c.supporting_evidence
    ]
    return "\n".join(lines) if lines else "(none)"


def render_retrieved_context(chunks: tuple[RetrievalChunkSnapshot, ...]) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(
        f"[{c.section_path}] (rank {c.rank}, pages {c.source_pages}): {c.text}" for c in chunks
    )


def render_citation_resolutions(resolutions: tuple[CitationResolution, ...]) -> str:
    """Renders each citation's DETERMINISTICALLY resolved logical target and
    its full allowed evidence SCOPE (cited section + every retrieved
    descendant, per the adopted hierarchy policy) so the Judge never has to
    infer section hierarchy itself from raw identifiers -- see
    heinzy.eval.citation_resolution.build_logical_section_index /
    resolve_citations."""
    if not resolutions:
        return "(no citations detected in the response)"
    lines = []
    for r in resolutions:
        line = f'- raw citation: "{r.raw_citation}" -> status: {r.resolution_status}'
        if r.resolution_status == RESOLVED:
            line += (
                f'; resolved_section: "{r.resolved_section_path}"'
                f'; allowed_evidence_scope_sections: {list(r.resolved_scope_section_ids)}'
                f'; allowed_evidence_scope_text: "{r.resolved_scope_retrieved_text}"'
            )
        elif r.resolution_status == AMBIGUOUS:
            line += f"; ambiguous_candidates: {list(r.ambiguous_candidates)}"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# judge_prompt_v1.txt loading + rendering (never duplicated inline)
# --------------------------------------------------------------------------- #
_VARIANT_HEADER_RE = re.compile(r"=+\nVARIANT [AB]: [A-Z ]+ ROW\n=+\n")
_SYS_MARKER = "--- SYSTEM ---"
_INPUT_MARKER = "--- INPUT TEMPLATE ---"
_PLACEHOLDER_RE = re.compile(r"\{\{[a-z_]+\}\}")


@dataclass(frozen=True)
class JudgePromptVariants:
    system_answerable: str
    user_template_answerable: str
    system_unanswerable: str
    user_template_unanswerable: str


def _split_variant_block(block: str, variant_name: str) -> tuple[str, str]:
    if _SYS_MARKER not in block:
        raise ValueError(f"{variant_name}: missing {_SYS_MARKER!r} marker")
    _, after_sys = block.split(_SYS_MARKER, 1)
    if _INPUT_MARKER not in after_sys:
        raise ValueError(f"{variant_name}: missing {_INPUT_MARKER!r} marker")
    system_text, user_text = after_sys.split(_INPUT_MARKER, 1)
    return system_text.strip("\n").rstrip(), (_INPUT_MARKER + user_text).strip("\n").rstrip()


def load_judge_prompt_variants(path: str | Path) -> JudgePromptVariants:
    """Deterministically split judge_prompt_v1.txt into its ANSWERABLE and
    UNANSWERABLE variants. Never rewrites the rubric/prompt text -- only
    locates the two documented delimiter markers and slices around them."""
    text = Path(path).read_text(encoding="utf-8")
    parts = _VARIANT_HEADER_RE.split(text)
    if len(parts) != 3:
        raise ValueError(
            f"expected exactly 2 'VARIANT A/B' headers in {path}, "
            f"found {len(parts) - 1}"
        )
    _, variant_a_block, variant_b_block = parts
    sys_a, user_a = _split_variant_block(variant_a_block, "VARIANT A")
    sys_b, user_b = _split_variant_block(variant_b_block, "VARIANT B")
    return JudgePromptVariants(
        system_answerable=sys_a,
        user_template_answerable=user_a,
        system_unanswerable=sys_b,
        user_template_unanswerable=user_b,
    )


@dataclass(frozen=True)
class RenderedJudgeInput:
    system: str
    user: str


def render_judge_request(
    subject: EvaluationSubject, variants: JudgePromptVariants
) -> RenderedJudgeInput:
    """Builds the exact content sent to the Judge. Takes ONLY an
    EvaluationSubject -- structurally cannot see arm/verifier/human-label
    data because those never exist on that type."""
    citation_block = render_citation_resolutions(subject.citation_resolutions)
    context_block = render_retrieved_context(subject.retrieved_chunks)

    if subject.answerable:
        user = variants.user_template_answerable
        user = user.replace("{{question}}", subject.question)
        user = user.replace("{{generated_response}}", subject.generated_response)
        user = user.replace("{{gold_answer}}", subject.gold_answer or "")
        user = user.replace("{{gold_claims}}", render_gold_claims(subject.gold_claims or ()))
        user = user.replace(
            "{{supporting_evidence}}", render_supporting_evidence(subject.gold_claims or ())
        )
        user = user.replace("{{retrieved_context}}", context_block)
        user = user.replace("{{cited_sections}}", citation_block)
        system = variants.system_answerable
    else:
        user = variants.user_template_unanswerable
        user = user.replace("{{question}}", subject.question)
        user = user.replace("{{generated_response}}", subject.generated_response)
        user = user.replace("{{unanswerable_reason}}", subject.unanswerable_reason or "")
        user = user.replace("{{retrieved_context}}", context_block)
        user = user.replace("{{cited_sections}}", citation_block)
        system = variants.system_unanswerable

    leftover = _PLACEHOLDER_RE.findall(user) + _PLACEHOLDER_RE.findall(system)
    if leftover:
        raise ValueError(f"unrendered placeholder(s) left in judge input: {leftover}")

    return RenderedJudgeInput(system=system, user=user)


# --------------------------------------------------------------------------- #
# judge design provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeDesignProvenance:
    judge_prompt_version: str
    judge_prompt_fingerprint: str
    judge_rubric_version: str
    judge_rubric_fingerprint: str
    judge_output_schema_version: str
    judge_output_schema_fingerprint: str
    evaluation_gold_version: str
    evaluation_gold_fingerprint: str


def compute_design_provenance(
    *,
    prompt_path: str | Path,
    rubric_path: str | Path,
    schema_path: str | Path,
    gold_path: str | Path,
    evaluation_gold_version: str,
    judge_prompt_version: str = JUDGE_PROMPT_VERSION,
    judge_rubric_version: str = JUDGE_RUBRIC_VERSION,
) -> JudgeDesignProvenance:
    """judge_prompt_version/judge_rubric_version default to the module's own
    "v1" constants for backward compatibility, but MUST be overridden by the
    caller whenever prompt_path/rubric_path point at a different versioned
    candidate (e.g. judge_prompt_v1.1.txt) -- otherwise the recorded
    provenance would silently misrepresent which prompt/rubric text actually
    produced a result, defeating the point of versioning a new candidate at
    all. The fingerprint is always computed from the actual file content
    regardless of this label, so a mismatched label is still detectable by
    fingerprint comparison, but callers should not rely on that as a
    substitute for passing the correct version string."""
    return JudgeDesignProvenance(
        judge_prompt_version=judge_prompt_version,
        judge_prompt_fingerprint=sha256_file(prompt_path),
        judge_rubric_version=judge_rubric_version,
        judge_rubric_fingerprint=sha256_file(rubric_path),
        judge_output_schema_version=JUDGE_OUTPUT_SCHEMA_VERSION,
        judge_output_schema_fingerprint=sha256_file(schema_path),
        evaluation_gold_version=evaluation_gold_version,
        evaluation_gold_fingerprint=sha256_file(gold_path),
    )


# --------------------------------------------------------------------------- #
# output schema loading / parsing / validation
# --------------------------------------------------------------------------- #
def load_output_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_judge_output(raw_text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """STRICT JSON only. The only permitted cleanup is stripping harmless
    surrounding whitespace -- no brace-extraction from prose, no field
    invention, no enum reinterpretation. Exactly one of the return values is
    None."""
    text = (raw_text or "").strip()
    if not text:
        return None, "empty judge output"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, f"judge output must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


def build_provider_schema(schema_doc: dict[str, Any], *, answerable: bool) -> dict[str, Any]:
    """Selects judge_output_schema_v1.json's answerable_schema or
    unanswerable_schema -- ONLY from the gold row's `answerable` flag, never
    from anything the response itself says -- and returns a SELF-CONTAINED
    copy: the selected sub-schema plus its own copy of the document's
    top-level `$defs`.

    Both `#/$defs/...` refs inside the sub-schema (e.g.
    `#/$defs/semantic_label_with_na`) resolve against whatever document is
    passed to a validator/generator, and `$defs` itself lives only at
    schema_doc's top level -- not inside either sub-schema. Without
    attaching it, a caller given only `schema_doc["answerable_schema"]`
    would have dangling refs. This is used for BOTH local validation
    (validate_judge_output) and, now, as the exact schema handed to a
    provider's structured-output constraint (see HTTPJudgeClient.run's
    `response_schema`) -- one function, one selection rule, no drift
    between what is sent to the model and what is validated afterward.

    Deep-copied so a caller mutating the result (or a provider serializing
    it into a request body) can never affect judge_output_schema_v1.json's
    loaded in-memory representation. Every required/additionalProperties/
    enum/allOf-if-then/not constraint from the frozen schema is preserved
    exactly -- nothing is loosened or reshaped.
    """
    schema_key = "answerable_schema" if answerable else "unanswerable_schema"
    schema = copy.deepcopy(schema_doc[schema_key])
    schema["$defs"] = copy.deepcopy(schema_doc.get("$defs", {}))
    return schema


def validate_judge_output(
    parsed: dict[str, Any], *, answerable: bool, schema_doc: dict[str, Any]
) -> list[str]:
    """Validates against judge_output_schema_v1.json's answerable_schema or
    unanswerable_schema, selected ONLY from the gold answerable flag. Returns
    every schema-validation error message found (empty list = valid)."""
    combined = build_provider_schema(schema_doc, answerable=answerable)
    validator_cls = jsonschema.validators.validator_for(combined)
    validator_cls.check_schema(combined)
    validator = validator_cls(combined)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(map(str, e.path)))
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


# --------------------------------------------------------------------------- #
# Ollama endpoint resolution (centralized -- fixes the bare "/api/chat" bug:
# an empty/unset endpoint must never reach requests.post as a URL)
# --------------------------------------------------------------------------- #
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"


def resolve_ollama_endpoint(explicit_endpoint: str | None) -> str:
    """Precedence: explicit endpoint > OLLAMA_HOST env var (non-empty) >
    DEFAULT_OLLAMA_ENDPOINT. Always returns a non-empty, trailing-slash-free
    URL -- an Ollama judge_client must never be constructed with an empty
    endpoint, which previously produced `MissingSchema: Invalid URL
    '/api/chat'`."""
    if explicit_endpoint and explicit_endpoint.strip():
        return explicit_endpoint.strip().rstrip("/")
    env_host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if env_host:
        return env_host.rstrip("/")
    return DEFAULT_OLLAMA_ENDPOINT


# --------------------------------------------------------------------------- #
# Judge reasoning/"thinking" configuration -- the ONE place model-name-based
# reasoning heuristics live. Temperature and thinking are independent
# configuration dimensions (never coupled): resolving "auto" thinking never
# touches temperature, and vice versa.
# --------------------------------------------------------------------------- #
THINK_AUTO = "auto"
THINK_ON = "on"
THINK_OFF = "off"
THINK_LOW = "low"
THINK_MEDIUM = "medium"
THINK_HIGH = "high"
_VALID_THINK_REQUESTS = (THINK_AUTO, THINK_ON, THINK_OFF, THINK_LOW, THINK_MEDIUM, THINK_HIGH)


@dataclass(frozen=True)
class ModelReasoningProfile:
    family: str
    pattern: re.Pattern[str]
    # Concrete requests (never "auto") this family accepts.
    supported_requests: tuple[str, ...]
    # What "auto" resolves to for this family -- must be in supported_requests.
    auto_request: str
    # Resolved request -> the exact value sent as Ollama's "think" field.
    think_value_map: dict[str, bool | str]


# Add a new profile here (never a scattered if/elif on model name elsewhere)
# to support another reasoning-capable model family.
_REASONING_PROFILES: tuple[ModelReasoningProfile, ...] = (
    ModelReasoningProfile(
        family="qwen",
        pattern=re.compile(r"^qwen", re.IGNORECASE),
        supported_requests=(THINK_ON, THINK_OFF),
        auto_request=THINK_ON,
        think_value_map={THINK_ON: True, THINK_OFF: False},
    ),
    ModelReasoningProfile(
        family="gpt-oss",
        pattern=re.compile(r"^gpt-oss", re.IGNORECASE),
        # Deliberately no THINK_OFF: do not silently send think=false to
        # GPT-OSS, and fail loudly (not a silent no-op) if a caller asks
        # for it -- see resolve_think_config's "not supported" branch.
        supported_requests=(THINK_LOW, THINK_MEDIUM, THINK_HIGH),
        auto_request=THINK_MEDIUM,
        think_value_map={THINK_LOW: "low", THINK_MEDIUM: "medium", THINK_HIGH: "high"},
    ),
)


def _find_reasoning_profile(judge_model: str) -> ModelReasoningProfile | None:
    for profile in _REASONING_PROFILES:
        if profile.pattern.match(judge_model):
            return profile
    return None


@dataclass(frozen=True)
class ThinkResolution:
    requested: str
    # None when thinking configuration does not apply to this provider
    # (e.g. azure_openai) -- distinct from a resolved concrete value.
    effective: str | None
    ollama_think_value: bool | str | None
    model_family: str | None


def resolve_think_config(
    judge_provider: str, judge_model: str, think_request: str
) -> ThinkResolution:
    """Centralized, testable, model-aware reasoning resolution. Callers MUST
    call this -- and let it raise -- BEFORE making any HTTP request; it is
    never called from inside a try/except that would downgrade a bad
    configuration into a per-row EXECUTION_ERROR.

    Raises ValueError for: an invalid think_request string, an
    unregistered/unknown model family on provider="ollama", or a concrete
    (non-"auto") request that family doesn't support (e.g. GPT-OSS +
    "off", or a non-GPT-OSS model + "low"/"medium"/"high").
    """
    if think_request not in _VALID_THINK_REQUESTS:
        raise ValueError(
            f"invalid judge_think={think_request!r}; must be one of {_VALID_THINK_REQUESTS}"
        )
    if judge_provider != "ollama":
        # Thinking/reasoning configuration is an Ollama-model concept in
        # this project's scope. Other providers (e.g. azure_openai) get an
        # explicit not-applicable resolution -- never a silently-wrong
        # Ollama-shaped "think" value applied to a different provider.
        return ThinkResolution(
            requested=think_request, effective=None, ollama_think_value=None, model_family=None
        )

    profile = _find_reasoning_profile(judge_model)
    if profile is None:
        raise ValueError(
            f"no reasoning profile registered for judge_model={judge_model!r}. "
            "Register a ModelReasoningProfile for this model family "
            "(_REASONING_PROFILES) before using it as an Ollama Judge model."
        )
    effective = profile.auto_request if think_request == THINK_AUTO else think_request
    if effective not in profile.supported_requests:
        raise ValueError(
            f"judge_model={judge_model!r} (family={profile.family!r}) does not support "
            f"judge_think={effective!r}. Supported for this family: {profile.supported_requests}."
        )
    return ThinkResolution(
        requested=think_request,
        effective=effective,
        ollama_think_value=profile.think_value_map[effective],
        model_family=profile.family,
    )


# --------------------------------------------------------------------------- #
# Judge model interface (isolated -- never the RAG Generator)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    judge_provider: str
    judge_model: str
    # Development default only -- a low-variance starting point, NOT a
    # requirement for the Control condition, NOT a guarantee of
    # deterministic output, and NOT necessarily the final tuned value. May
    # later be changed using DEVELOPMENT data only (see module docs).
    temperature: float = 0.0
    judge_think: str = THINK_AUTO
    timeout: float | None = None


@dataclass(frozen=True)
class JudgeCallResult:
    raw_text: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model_call_made: bool
    latency_seconds: float
    model_tag: str
    # True iff the provider response carried a separate reasoning/thinking
    # field alongside message.content. The trace's TEXT is never stored or
    # forwarded anywhere (no engineering need for it yet, and it would bloat
    # result files) -- only this presence flag.
    thinking_trace_present: bool = False


class JudgeClient(Protocol):
    def run(
        self, system_prompt: str, user_prompt: str, *, temperature: float,
        think_value: bool | str | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> JudgeCallResult: ...


class HTTPJudgeClient:
    """Real, isolated Judge model-call interface.

    Deliberately independent of heinzy.generation.generator.Generator, which
    carries RAG-specific concerns (abstention policy, tool loop, citation
    extraction, refusal text) that have nothing to do with grading. Reuses
    only: (a) the same provider-dispatch pattern (MODEL_PROVIDER-style
    ollama/azure_openai branching) and (b) extract_usage(), which is already
    a generic, provider-branching parsing function with no RAG-specific
    content.

    NOT exercised by this task or by any test in tests/test_judge.py --
    tests use a scripted fake JudgeClient exclusively. No real Judge/API
    call is made anywhere in this implementation.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        endpoint: str | None = None,
        azure_api_key: str | None = None,
        azure_api_version: str = "2024-10-21",
    ) -> None:
        self.provider = provider
        self.model = model
        if provider == "ollama":
            # Precedence: explicit endpoint > OLLAMA_HOST > 127.0.0.1:11434.
            # Never empty -- an empty endpoint previously produced
            # `MissingSchema: Invalid URL '/api/chat'`.
            self.endpoint = resolve_ollama_endpoint(endpoint)
        else:
            # Azure/OpenAI-style providers: never silently apply the Ollama
            # default here. An explicit endpoint is this provider's own
            # contract, unrelated to OLLAMA_HOST/127.0.0.1.
            self.endpoint = (endpoint or "").rstrip("/")
        self.azure_api_key = azure_api_key or ""
        self.azure_api_version = azure_api_version

    def run(
        self, system_prompt: str, user_prompt: str, *, temperature: float,
        think_value: bool | str | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> JudgeCallResult:
        """response_schema, when given, is a SELF-CONTAINED JSON Schema
        (already selected answerable/unanswerable + its own $defs -- see
        build_provider_schema) that the orchestration layer chose from the
        gold row's answerable flag. This client never loads gold, never
        infers answerability, and never chooses a schema itself -- it only
        forwards whatever schema it is given as the provider's structured-
        output constraint. Passing structured output through Ollama's
        `format` is an ADDITIONAL generation constraint; it never replaces
        the strict local JSON parse + schema validation that still runs
        downstream on message.content."""
        import requests  # local import: keep this optional dependency out of
        # the hot path for callers (tests, dry-run, inspection) that never
        # instantiate a real HTTP client.

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        t0 = time.monotonic()
        thinking_trace_present = False
        if self.provider == "ollama":
            payload: dict[str, Any] = {
                "model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "seed": 0},
            }
            # Only set "think" when the caller resolved a concrete value for
            # this model (see resolve_think_config) -- never a hard-coded
            # universal True/False that would be wrong for some family.
            if think_value is not None:
                payload["think"] = think_value
            # Structured-output generation constraint. The schema itself is
            # never modified/loosened here -- forwarded exactly as built by
            # build_provider_schema.
            if response_schema is not None:
                payload["format"] = response_schema
            resp = requests.post(
                f"{self.endpoint}/api/chat", json=payload, timeout=timeout or 180,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message") or {}
            # A thinking-capable model may return reasoning separately from
            # the final answer (commonly message["thinking"]). Only its
            # PRESENCE is recorded; its text is never read into raw_text and
            # never stored.
            thinking_trace_present = bool(message.get("thinking"))
        elif self.provider == "azure_openai":
            # Azure/OpenAI structured-output support is not implemented in
            # this patch -- response_schema is intentionally never read or
            # forwarded on this branch, so it cannot leak Ollama's `format`
            # shape into an unrelated provider's request.
            url = (
                f"{self.endpoint}/openai/deployments/{self.model}"
                f"/chat/completions?api-version={self.azure_api_version}"
            )
            resp = requests.post(
                url,
                json={"messages": messages, "temperature": temperature},
                headers={"api-key": self.azure_api_key, "Content-Type": "application/json"},
                timeout=timeout or 180,
            )
            resp.raise_for_status()
            raw = resp.json()
            message = raw["choices"][0]["message"]
            data = {"message": message, "raw": raw}
        else:
            raise ValueError(
                f"Unknown judge_provider={self.provider!r}. Supported: ollama, azure_openai."
            )
        latency = time.monotonic() - t0

        # message.content ONLY -- never concatenated with a thinking/
        # reasoning field, per the strict-JSON-parsing contract downstream.
        raw_text = message.get("content") or ""
        usage = extract_usage(self.provider, data)
        return JudgeCallResult(
            raw_text=raw_text,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            model_call_made=True,
            latency_seconds=latency,
            model_tag=self.model,
            thinking_trace_present=thinking_trace_present,
        )


# --------------------------------------------------------------------------- #
# result record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeResultRecord:
    """A future human-label calibration join must use `(question_id, arm)`
    as the ROW IDENTITY key -- both fields are carried here specifically so
    that join can be made -- and treat `evaluation_subject_fingerprint` as a
    separate ARTIFACT INTEGRITY check performed after that join (does this
    Judge result's fingerprint match what the human label was recorded
    against?), never as the join key itself. Control and Treatment can
    legitimately share an identical generated_response/retrieved_context/
    citation mapping (their `evaluation_subject_fingerprint` values are then
    equal by construction), so a fingerprint-only join would silently
    conflate two distinct rows. See the module docstring's ROW IDENTITY /
    ARTIFACT INTEGRITY section for the full contract.
    """

    # orchestration / subject metadata
    question_id: str
    response_id: str | None
    arm: str
    answerable: bool
    source_experiment_file: str
    source_record_index: int

    # subject fingerprints -- evaluation_subject_fingerprint is an artifact
    # integrity check, NOT a row-identity key (see class docstring above).
    generated_response_fingerprint: str
    retrieved_context_fingerprint: str
    evaluation_subject_fingerprint: str

    # judge design provenance
    judge_prompt_version: str
    judge_prompt_fingerprint: str
    judge_rubric_version: str
    judge_rubric_fingerprint: str
    judge_output_schema_version: str
    judge_output_schema_fingerprint: str
    evaluation_gold_version: str
    evaluation_gold_fingerprint: str
    rendered_judge_input_fingerprint: str

    # model config -- this is a FULL JUDGE CONFIGURATION comparison, not a
    # bare model-name comparison: two runs with the same judge_model but
    # different temperature/thinking settings are different configurations.
    judge_provider: str
    judge_model: str
    # "requested" is what ModelConfig asked for; "effective" is what was
    # actually used for this call. In this implementation temperature is
    # never silently altered (auto reasoning selection never changes it),
    # so requested == effective always -- both are still recorded
    # separately so that invariant is auditable rather than assumed.
    judge_temperature_requested: float
    judge_temperature_effective: float
    # judge_think_requested is the raw configured value (e.g. "auto");
    # judge_think_effective is what that resolved to for this judge_model
    # (e.g. "on" for a qwen model, "medium" for a gpt-oss model), or None
    # when thinking configuration does not apply to this provider.
    judge_think_requested: str
    judge_think_effective: str | None
    timeout_seconds: float | None

    # execution
    timestamp: str
    latency_ms: float | None
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    usage_total_tokens: int | None
    usage_model_call_made: bool
    # Whether the provider response carried a separate reasoning/thinking
    # field alongside the final answer. None when no call was made at all
    # (EXECUTION_ERROR before any response existed). The trace TEXT itself
    # is never stored here or anywhere else.
    thinking_trace_present: bool | None

    # citation resolution (structured mapping from section 9)
    citation_resolutions: tuple[dict[str, Any], ...]

    # output
    execution_status: str
    raw_judge_output: str | None
    parsed_judge_output: dict[str, Any] | None
    error_type: str | None
    error_message: str | None

    # optional, for auditability -- never excessively duplicative at this
    # scale (60 rows), and never a substitute for the mandatory fingerprint.
    rendered_judge_input: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.execution_status not in _VALID_EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status must be one of {_VALID_EXECUTION_STATUSES}, "
                f"got {self.execution_status!r}"
            )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def evaluate_subject(
    subject: EvaluationSubject,
    variants: JudgePromptVariants,
    judge_client: JudgeClient,
    *,
    model_config: ModelConfig,
    design_provenance: JudgeDesignProvenance,
    schema_doc: dict[str, Any],
    source_experiment_file: str,
    source_record_index: int,
    arm: str,
    response_id: str | None = None,
    store_rendered_input: bool = True,
) -> JudgeResultRecord:
    """Grades exactly ONE subject with exactly ONE independent Judge call.
    Never receives, stores, or forwards a prior Judge output or any other
    subject's data."""
    # Resolve thinking/reasoning configuration BEFORE any HTTP request, and
    # OUTSIDE the try/except below: an invalid model/reasoning combination
    # must raise loudly here, never be silently downgraded to a per-row
    # EXECUTION_ERROR. Temperature is untouched by this resolution -- the
    # two configuration dimensions are independent (see ModelConfig).
    think_resolution = resolve_think_config(
        model_config.judge_provider, model_config.judge_model, model_config.judge_think
    )

    rendered = render_judge_request(subject, variants)
    rendered_fp = sha256_text(rendered.system + "\x1e" + rendered.user)

    # The SAME selection rule (answerable flag only) and the SAME helper
    # used for local post-hoc validation build the provider-facing
    # structured-output schema, so there is no chance of the two drifting
    # apart -- see build_provider_schema.
    provider_schema = build_provider_schema(schema_doc, answerable=subject.answerable)

    execution_status: str
    raw_output: str | None = None
    parsed_output: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float | None = None
    usage_input = usage_output = usage_total = None
    usage_call_made = False
    thinking_trace_present: bool | None = None

    try:
        call_result = judge_client.run(
            rendered.system, rendered.user,
            temperature=model_config.temperature,
            think_value=think_resolution.ollama_think_value,
            response_schema=provider_schema,
            timeout=model_config.timeout,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
        # failure to obtain a Judge response at all is EXECUTION_ERROR,
        # regardless of the underlying exception type (network, HTTP,
        # provider, timeout, ...).
        execution_status = EXECUTION_ERROR
        error_type = type(exc).__name__
        error_message = str(exc)
    else:
        latency_ms = call_result.latency_seconds * 1000.0
        raw_output = call_result.raw_text
        usage_input = call_result.input_tokens
        usage_output = call_result.output_tokens
        usage_total = call_result.total_tokens
        usage_call_made = call_result.model_call_made
        thinking_trace_present = call_result.thinking_trace_present

        parsed, parse_err = parse_judge_output(raw_output)
        if parse_err is not None:
            execution_status = PARSE_ERROR
            error_type = "JSONDecodeError"
            error_message = parse_err
        else:
            schema_errors = validate_judge_output(
                parsed, answerable=subject.answerable, schema_doc=schema_doc
            )
            if schema_errors:
                execution_status = SCHEMA_ERROR
                error_type = "SchemaValidationError"
                error_message = "; ".join(schema_errors)
            else:
                execution_status = EXECUTION_SUCCESS
                parsed_output = parsed

    return JudgeResultRecord(
        question_id=subject.question_id,
        response_id=response_id,
        arm=arm,
        answerable=subject.answerable,
        source_experiment_file=source_experiment_file,
        source_record_index=source_record_index,
        generated_response_fingerprint=sha256_text(subject.generated_response),
        retrieved_context_fingerprint=fingerprint_retrieved_context(subject.retrieved_chunks),
        evaluation_subject_fingerprint=evaluation_subject_fingerprint(subject),
        judge_prompt_version=design_provenance.judge_prompt_version,
        judge_prompt_fingerprint=design_provenance.judge_prompt_fingerprint,
        judge_rubric_version=design_provenance.judge_rubric_version,
        judge_rubric_fingerprint=design_provenance.judge_rubric_fingerprint,
        judge_output_schema_version=design_provenance.judge_output_schema_version,
        judge_output_schema_fingerprint=design_provenance.judge_output_schema_fingerprint,
        evaluation_gold_version=design_provenance.evaluation_gold_version,
        evaluation_gold_fingerprint=design_provenance.evaluation_gold_fingerprint,
        rendered_judge_input_fingerprint=rendered_fp,
        judge_provider=model_config.judge_provider,
        judge_model=model_config.judge_model,
        judge_temperature_requested=model_config.temperature,
        judge_temperature_effective=model_config.temperature,
        judge_think_requested=think_resolution.requested,
        judge_think_effective=think_resolution.effective,
        timeout_seconds=model_config.timeout,
        timestamp=_utc_timestamp(),
        latency_ms=latency_ms,
        usage_input_tokens=usage_input,
        usage_output_tokens=usage_output,
        usage_total_tokens=usage_total,
        usage_model_call_made=usage_call_made,
        thinking_trace_present=thinking_trace_present,
        citation_resolutions=tuple(
            _citation_resolution_to_dict(r) for r in subject.citation_resolutions
        ),
        execution_status=execution_status,
        raw_judge_output=raw_output,
        parsed_judge_output=parsed_output,
        error_type=error_type,
        error_message=error_message,
        rendered_judge_input=(
            {"system": rendered.system, "user": rendered.user} if store_rendered_input else None
        ),
    )
