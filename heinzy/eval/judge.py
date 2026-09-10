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

import hashlib
import json
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
    if not resolutions:
        return "(no citations detected in the response)"
    lines = []
    for r in resolutions:
        line = f'- raw citation: "{r.raw_citation}" -> status: {r.resolution_status}'
        if r.resolution_status == RESOLVED:
            line += (
                f'; resolved_section: "{r.resolved_section_path}"; '
                f'resolved_retrieved_text: "{r.resolved_retrieved_text}"'
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
) -> JudgeDesignProvenance:
    return JudgeDesignProvenance(
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        judge_prompt_fingerprint=sha256_file(prompt_path),
        judge_rubric_version=JUDGE_RUBRIC_VERSION,
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


def validate_judge_output(
    parsed: dict[str, Any], *, answerable: bool, schema_doc: dict[str, Any]
) -> list[str]:
    """Validates against judge_output_schema_v1.json's answerable_schema or
    unanswerable_schema, selected ONLY from the gold answerable flag. Returns
    every schema-validation error message found (empty list = valid)."""
    schema_key = "answerable_schema" if answerable else "unanswerable_schema"
    schema = schema_doc[schema_key]
    # $ref pointers inside the sub-schema ("#/$defs/...") resolve against
    # the schema document passed to the validator, so $defs (which lives at
    # schema_doc's top level, not inside either sub-schema) must be attached
    # to whatever we validate against.
    combined = {**schema, "$defs": schema_doc.get("$defs", {})}
    validator_cls = jsonschema.validators.validator_for(combined)
    validator_cls.check_schema(combined)
    validator = validator_cls(combined)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(map(str, e.path)))
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


# --------------------------------------------------------------------------- #
# Judge model interface (isolated -- never the RAG Generator)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    judge_provider: str
    judge_model: str
    temperature: float = 0.0
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


class JudgeClient(Protocol):
    def run(
        self, system_prompt: str, user_prompt: str, *, temperature: float,
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
        self.endpoint = (endpoint or "").rstrip("/")
        self.azure_api_key = azure_api_key or ""
        self.azure_api_version = azure_api_version

    def run(
        self, system_prompt: str, user_prompt: str, *, temperature: float,
        timeout: float | None = None,
    ) -> JudgeCallResult:
        import requests  # local import: keep this optional dependency out of
        # the hot path for callers (tests, dry-run, inspection) that never
        # instantiate a real HTTP client.

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        t0 = time.monotonic()
        if self.provider == "ollama":
            resp = requests.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": self.model, "messages": messages, "stream": False,
                    "think": False, "options": {"temperature": temperature, "seed": 0},
                },
                timeout=timeout or 180,
            )
            resp.raise_for_status()
            data = resp.json()
        elif self.provider == "azure_openai":
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
            data = {"message": raw["choices"][0]["message"], "raw": raw}
        else:
            raise ValueError(
                f"Unknown judge_provider={self.provider!r}. Supported: ollama, azure_openai."
            )
        latency = time.monotonic() - t0

        raw_text = (data.get("message") or {}).get("content") or ""
        usage = extract_usage(self.provider, data)
        return JudgeCallResult(
            raw_text=raw_text,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            model_call_made=True,
            latency_seconds=latency,
            model_tag=self.model,
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

    # model config
    judge_provider: str
    judge_model: str
    temperature: float
    timeout_seconds: float | None

    # execution
    timestamp: str
    latency_ms: float | None
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    usage_total_tokens: int | None
    usage_model_call_made: bool

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
    rendered = render_judge_request(subject, variants)
    rendered_fp = sha256_text(rendered.system + "\x1e" + rendered.user)

    execution_status: str
    raw_output: str | None = None
    parsed_output: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float | None = None
    usage_input = usage_output = usage_total = None
    usage_call_made = False

    try:
        call_result = judge_client.run(
            rendered.system, rendered.user,
            temperature=model_config.temperature, timeout=model_config.timeout,
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
        temperature=model_config.temperature,
        timeout_seconds=model_config.timeout,
        timestamp=_utc_timestamp(),
        latency_ms=latency_ms,
        usage_input_tokens=usage_input,
        usage_output_tokens=usage_output,
        usage_total_tokens=usage_total,
        usage_model_call_made=usage_call_made,
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
