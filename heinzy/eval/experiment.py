"""
Paired Control/Treatment pilot experiment runner (execution/data-collection
only -- no scoring, no GAS, no judge, no statistics; see module docstring of
scripts/run_pilot_experiment.py).

Control:
    question -> retrieval -> Generator.generate() -> final answer
    (unchanged by v2.1 -- see Treatment strategy below)

Treatment strategy v2.1 (NOT purely a prompt change -- this is a routing
policy on top of Verifier Prompt v2, added after the real v2 pilot showed
the verifier fabricating facts/citations on zero-hit questions):

    question -> SAME cached retrieval
    IF len(hits) == 0:
        -> deterministic ABSTAIN. The verifier is never called: there is
           nothing retrieved for it to reason about, so there is no call
           whose output could need checking. Treatment final_text is the
           SAME standardized refusal text Control already produced via
           Layer 1 (draft.text), not new prose. verifier_call_made=False,
           zero_hit_guard_triggered=True.
    IF len(hits) > 0:
        -> SAME Generator.generate() draft -> one verify_answer() pass
           (Verifier Prompt v2) -> KEEP / REVISE / ABSTAIN -> final answer.
           verifier_call_made=True, zero_hit_guard_triggered=False.

In other words, Treatment is no longer "always one additional verifier
call" -- it is "zero-hit retrieval -> deterministic abstention (zero
additional model calls); nonzero-hit retrieval -> one verifier pass",
still never more than one additional model call in either branch. For
nonzero-hit questions, Verifier Prompt v2 is unchanged and still the sole
judge of whether evidence is sufficient -- this guard does not do
evidence-sufficiency scoring, reranking, or threshold changes; it only
short-circuits the one case where zero retrieved chunks makes a verifier
opinion structurally meaningless.

pre:  rows (heinzy.eval.dataset.Row, already loaded + validated), a
      Retriever and Generator built from the SAME config/store the caller
      wants measured, and (optionally) a distinct verifier_generator.
post: run_experiment() returns one ExperimentResult per row, each carrying
      a full retrieval snapshot, the Control ArmResult, and the
      TreatmentResult -- never a gold_answer/gold_claims/answerable field
      (join those back in later, by question_id, at grading time).
invariant: NON-NEGOTIABLE per the experiment spec --
  - retrieval happens exactly once per row; the same `hits` list object is
    passed to both Generator.generate() and (when len(hits) > 0)
    verify_answer().
  - Generator.generate() is called exactly once per row; its Answer is both
    the Control final answer and the Treatment draft (never called twice).
  - verify_answer() is called AT MOST once per row, and only when
    len(hits) > 0 -- including when the draft already refused via Layer 2
    sentinel (a nonzero-hit refusal is still verified) -- and is given only
    (question, draft, hits, verifier_generator), never `row`. Zero-hit rows
    never call it at all (v2.1's guard), so there is still never a second
    or third model call in any path.
invariant: NO GOLD LEAKAGE. This module imports Row to read `.question`/
           `.id`/`.question_type`/`.question_family_id`/`.policy_unit_id`
           for bookkeeping, but never passes the Row object itself, and
           never reads/forwards `.answerable`, `.gold_answer`,
           `.gold_claims`, or `.unanswerable_reason` anywhere. The zero-hit
           guard's routing decision depends on len(hits) alone -- no gold or
           evaluator field is ever consulted to decide whether to skip the
           verifier.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from heinzy.eval.dataset import Row, ValidationError, load_dataset, validate_dataset
from heinzy.generation.generator import Answer, Generator
from heinzy.generation.verify import (
    DECISION_ABSTAIN,
    VERIFIER_PROMPT_VERSION,
    VerificationResult,
    verifier_prompt_fingerprint,
    verify_answer,
)
from heinzy.retrieval.store import ScoredChunk

DEFAULT_EXECUTION_SEED = 0

# Treatment *strategy* version -- distinct from VERIFIER_PROMPT_VERSION.
# v2.1 = Verifier Prompt v2 + the deterministic zero-hit guard below. Bump
# this when the ROUTING policy changes; bump VERIFIER_PROMPT_VERSION
# separately when the prompt text itself changes. The two are independent.
TREATMENT_STRATEGY_VERSION = "v2.1"

# Machine-readable reason recorded on TreatmentResult.verifier_reason when
# the zero-hit guard fires (verifier_reason otherwise holds the verifier's
# own free-text rationale, so this sentinel string must stay unambiguous).
ZERO_HIT_GUARD_REASON = "zero_hit_guard"

VerifierFn = Callable[[str, Answer, list[ScoredChunk], Generator], VerificationResult]


# --------------------------------------------------------------------------- #
# dataset loading gate (script responsibility, exposed here so it's testable
# without invoking the CLI)
# --------------------------------------------------------------------------- #
class DatasetValidationError(ValueError):
    """Raised by load_and_validate_rows(); .errors carries every ValidationError."""

    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        super().__init__(
            f"{len(errors)} validation error(s): "
            + "; ".join(str(e) for e in errors)
        )


def load_and_validate_rows(path: str | Path) -> list[Row]:
    """Load the QA gold dataset and validate it before any retrieval/model call.

    Raises DatasetValidationError (carrying every ValidationError, not just
    the first) rather than returning a partially-invalid dataset. Callers
    (the pilot CLI) must not construct a store/retriever/generator until
    this has returned successfully.
    """
    rows = load_dataset(path)
    errors = validate_dataset(rows)
    if errors:
        raise DatasetValidationError(errors)
    return rows


# --------------------------------------------------------------------------- #
# tool-disabling (experiment validity: rule 8)
# --------------------------------------------------------------------------- #
def ensure_tools_disabled(generator) -> bool:
    """Force tool-calling off on `generator` for this experiment.

    Returns True when an override was actually needed (i.e. the generator's
    *effective*, already-computed `use_tools` -- built from real config plus
    heinzy.governance.loader.governance_available(), not a default guess --
    was True), so the caller can warn loudly rather than silently changing
    behavior. Always leaves generator.use_tools False.
    """
    was_enabled = bool(getattr(generator, "use_tools", False))
    generator.use_tools = False
    return was_enabled


# --------------------------------------------------------------------------- #
# question execution order (rule 9: randomize question order only, never
# arm order -- arm order is inherently Control/shared-draft first, Treatment
# verifier second, because the draft is shared)
# --------------------------------------------------------------------------- #
def shuffled_order(n: int, seed: int = DEFAULT_EXECUTION_SEED) -> list[int]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


# --------------------------------------------------------------------------- #
# result schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrievalChunkSnapshot:
    rank: int
    chunk_id: str
    score: float
    doc_id: str
    section_path: str | None
    source_pages: list[int]
    text: str


@dataclass(frozen=True)
class RetrievalSnapshot:
    query: str
    k: int
    latency_seconds: float
    chunks: list[RetrievalChunkSnapshot]


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    # True iff a model call was actually made. Distinguishes "no call at all"
    # (e.g. a Layer 1 refusal -- token cost is truly 0) from "a call was
    # made but the provider's response didn't report usage" (token fields
    # are None because they are genuinely unknown, not zero). Required, no
    # default: every construction site must state this explicitly rather
    # than have it guessed from whether the token fields happen to be None.
    model_call_made: bool


@dataclass(frozen=True)
class ArmResult:
    text: str
    raw_text: str | None
    refused: bool
    refusal_reason: str | None
    cited_sections: list[str]
    unsupported_citations: list[str]
    model_tag: str
    usage: ModelUsage
    generation_latency_seconds: float


@dataclass(frozen=True)
class TreatmentResult:
    draft_text: str
    final_text: str
    refused: bool
    verifier_decision: str
    verifier_reason: str
    raw_verifier_output: str
    verifier_model_tag: str | None
    verifier_usage: ModelUsage
    verifier_latency_seconds: float
    effective_total_usage: ModelUsage
    effective_e2e_latency_seconds: float
    # v2.1 routing provenance. verifier_call_made is False only for the
    # zero-hit guard path; zero_hit_guard_triggered says WHY it's False
    # (currently the only reason it ever is False, but kept as a separate
    # explicit flag rather than inferred from verifier_call_made so a future
    # second skip-reason wouldn't have to overload this one).
    verifier_call_made: bool
    zero_hit_guard_triggered: bool
    # Beyond the suggested sketch: preserves *why* a verifier output fell
    # back to ABSTAIN (see heinzy.generation.verify's fail-safe invariant).
    # None for both a clean verifier parse AND the zero-hit guard path
    # (neither is a parse failure); set only on the verify.py fallback.
    parse_error: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    question_id: str
    question: str
    question_type: str
    question_family_id: str | None
    policy_unit_id: str | None
    retrieval: RetrievalSnapshot
    control: ArmResult
    # Beyond the suggested sketch: P1 explicitly requires
    # control_e2e_latency_seconds = retrieval + first pass to be recorded.
    control_e2e_latency_seconds: float
    treatment: TreatmentResult


@dataclass
class ExperimentReport:
    timestamp: str
    config_hash: str
    generator_model: str
    verifier_model: str
    verifier_prompt_version: str
    verifier_prompt_fingerprint: str
    # The Treatment ROUTING strategy (Verifier Prompt v2 + zero-hit guard),
    # independent of verifier_prompt_version -- see TREATMENT_STRATEGY_VERSION.
    treatment_strategy_version: str
    dataset_path: str
    dataset_row_count: int
    k: int
    backend: str
    tools_enabled: bool
    tools_originally_enabled: bool
    execution_seed: int | None
    results: list[ExperimentResult] = field(default_factory=list)


def _usage_from_dict(d: dict[str, int | None] | None) -> ModelUsage:
    """Build a ModelUsage from Answer.usage.

    Answer.usage is None exactly when no model call was made at all (a
    Layer 1 refusal returns before the model is ever reached) -- distinct
    from a call that WAS made but whose provider response didn't report
    usage, which is a dict with None-valued fields. That distinction has to
    survive into aggregation (see _sum_usage/_combine_usage_field below), so
    it is captured here as model_call_made rather than left to be inferred
    later from token values.
    """
    if d is None:
        return ModelUsage(
            input_tokens=None, output_tokens=None, total_tokens=None,
            model_call_made=False,
        )
    return ModelUsage(
        input_tokens=d.get("input_tokens"),
        output_tokens=d.get("output_tokens"),
        total_tokens=d.get("total_tokens"),
        model_call_made=True,
    )


def _combine_usage_field(
    a_value: int | None, a_call_made: bool, b_value: int | None, b_call_made: bool,
) -> int | None:
    """Combine one usage field across two calls for product-cost aggregation.

    A side with no call contributes 0 (nothing was spent). A side whose call
    WAS made but whose value is None contributes "unknown" -- and unknown
    plus anything is unknown, since a token count is never invented. Only
    when every side that actually ran reported a real number does this
    return a number.
    """
    a_effective = 0 if not a_call_made else a_value
    b_effective = 0 if not b_call_made else b_value
    if a_effective is None or b_effective is None:
        return None
    return a_effective + b_effective


def _sum_usage(a: ModelUsage, b: ModelUsage) -> ModelUsage:
    return ModelUsage(
        input_tokens=_combine_usage_field(
            a.input_tokens, a.model_call_made, b.input_tokens, b.model_call_made
        ),
        output_tokens=_combine_usage_field(
            a.output_tokens, a.model_call_made, b.output_tokens, b.model_call_made
        ),
        total_tokens=_combine_usage_field(
            a.total_tokens, a.model_call_made, b.total_tokens, b.model_call_made
        ),
        model_call_made=a.model_call_made or b.model_call_made,
    )


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_experiment(
    rows: list[Row],
    retriever,
    generator: Generator,
    *,
    verifier_generator: Generator | None = None,
    k: int | None = None,
    verifier: VerifierFn = verify_answer,
    execution_order: list[int] | None = None,
    on_result: Callable[[ExperimentResult], None] | None = None,
) -> list[ExperimentResult]:
    """Run the paired Control/Treatment pilot over `rows`.

    verifier_generator defaults to `generator` (same model for first pass
    and verification); pass a distinct instance to run verification against
    a different model tag. Results are returned in execution order (each
    carries `question_id`, so downstream joins by id, not position).
    """
    vgen = verifier_generator if verifier_generator is not None else generator
    order = execution_order if execution_order is not None else list(range(len(rows)))
    results: list[ExperimentResult] = []

    for idx in order:
        row = rows[idx]

        # 1. retrieval, exactly once.
        t0 = time.monotonic()
        retrieval_result = retriever.retrieve(row.question, k=k)
        retrieval_latency = time.monotonic() - t0
        hits = list(retrieval_result.hits)  # cached, reused identically below

        retrieval_snapshot = RetrievalSnapshot(
            query=row.question,
            k=retrieval_result.k,
            latency_seconds=retrieval_latency,
            chunks=[
                RetrievalChunkSnapshot(
                    rank=i + 1,
                    chunk_id=h.chunk_id,
                    score=h.score,
                    doc_id=h.doc_id,
                    section_path=h.section_path,
                    source_pages=list(h.source_pages),
                    text=h.text,
                )
                for i, h in enumerate(hits)
            ],
        )

        # 2. first pass, exactly once -- this Answer is BOTH the Control
        # final answer and the Treatment draft. Never call generate() again.
        t0 = time.monotonic()
        draft = generator.generate(row.question, hits)
        first_pass_latency = time.monotonic() - t0

        control_usage = _usage_from_dict(draft.usage)
        control = ArmResult(
            text=draft.text,
            raw_text=draft.raw_text,
            refused=draft.refused,
            refusal_reason=draft.refusal_reason,
            cited_sections=list(draft.cited_sections),
            unsupported_citations=list(draft.unsupported_citations),
            model_tag=draft.model_tag,
            usage=control_usage,
            generation_latency_seconds=first_pass_latency,
        )
        control_e2e = retrieval_latency + first_pass_latency

        # 3. v2.1 routing: zero-hit questions never reach the verifier at
        # all -- there is nothing retrieved for it to check, so a verifier
        # opinion here is structurally meaningless and, per the real v2
        # pilot, a demonstrated fabrication risk. This decision depends on
        # len(hits) ONLY -- no gold/evaluator field is ever consulted here.
        if len(hits) == 0:
            verifier_usage = ModelUsage(
                input_tokens=None, output_tokens=None, total_tokens=None,
                model_call_made=False,
            )
            treatment = TreatmentResult(
                draft_text=draft.text,
                # Same standardized refusal text Control already produced
                # via Layer 1 -- not new prose, so this cannot introduce a
                # fabricated fact or citation by construction.
                final_text=draft.text,
                refused=True,
                verifier_decision=DECISION_ABSTAIN,
                verifier_reason=ZERO_HIT_GUARD_REASON,
                raw_verifier_output="",
                verifier_model_tag=None,
                verifier_usage=verifier_usage,
                verifier_latency_seconds=0.0,
                effective_total_usage=_sum_usage(control_usage, verifier_usage),
                effective_e2e_latency_seconds=control_e2e + 0.0,
                parse_error=None,
                verifier_call_made=False,
                zero_hit_guard_triggered=True,
            )
        else:
            # Verification, exactly once (rules 6 & 7: a refused draft --
            # here only a Layer 2 sentinel refusal is possible, since Layer 1
            # requires zero hits -- still goes through this). Only
            # question/draft/hits/vgen cross this call; `row` never does.
            verification = verifier(row.question, draft, hits, vgen)

            verifier_usage = ModelUsage(
                input_tokens=verification.input_tokens,
                output_tokens=verification.output_tokens,
                total_tokens=verification.total_tokens,
                model_call_made=verification.model_call_made,
            )
            treatment = TreatmentResult(
                draft_text=draft.text,
                final_text=verification.final_text,
                refused=verification.refused,
                verifier_decision=verification.decision,
                verifier_reason=verification.reason,
                raw_verifier_output=verification.raw_verifier_output,
                verifier_model_tag=verification.model_tag,
                verifier_usage=verifier_usage,
                verifier_latency_seconds=verification.latency_seconds,
                effective_total_usage=_sum_usage(control_usage, verifier_usage),
                effective_e2e_latency_seconds=control_e2e + verification.latency_seconds,
                parse_error=verification.parse_error,
                verifier_call_made=True,
                zero_hit_guard_triggered=False,
            )

        result = ExperimentResult(
            question_id=row.id,
            question=row.question,
            question_type=row.question_type,
            question_family_id=row.question_family_id,
            policy_unit_id=row.policy_unit_id,
            retrieval=retrieval_snapshot,
            control=control,
            control_e2e_latency_seconds=control_e2e,
            treatment=treatment,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)

    return results


def build_report(
    results: list[ExperimentResult],
    *,
    cfg,
    generator: Generator,
    dataset_path: str,
    dataset_row_count: int,
    k: int,
    backend: str,
    tools_originally_enabled: bool,
    execution_seed: int | None,
    verifier_generator: Generator | None = None,
) -> ExperimentReport:
    vgen = verifier_generator if verifier_generator is not None else generator
    return ExperimentReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        config_hash=cfg.config_hash,
        generator_model=generator.model_tag,
        verifier_model=vgen.model_tag,
        verifier_prompt_version=VERIFIER_PROMPT_VERSION,
        verifier_prompt_fingerprint=verifier_prompt_fingerprint(),
        treatment_strategy_version=TREATMENT_STRATEGY_VERSION,
        dataset_path=str(dataset_path),
        dataset_row_count=dataset_row_count,
        k=k,
        backend=backend,
        tools_enabled=False,
        tools_originally_enabled=tools_originally_enabled,
        execution_seed=execution_seed,
        results=results,
    )
