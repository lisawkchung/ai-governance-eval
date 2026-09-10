"""
Second-pass evidence verifier (Treatment arm of the paired pilot experiment).

pre:  question (str), draft (the SAME Answer object the Control arm
      received, produced by exactly one Generator.generate() call), hits
      (the SAME cached retrieval snapshot Control saw), and a generator used
      only for its chat/model plumbing.
post: VerificationResult carrying decision (KEEP|REVISE|ABSTAIN), the final
      answer text, and enough raw/usage/timing detail to audit and re-grade
      the call later.
invariant: NO GOLD DATA. verify_answer()'s signature is (question: str,
           draft: Answer, hits: list[ScoredChunk], generator: Generator) --
           there is no parameter through which a dataset Row, answerable,
           unanswerable_reason, gold_answer, gold_claims, supporting_evidence,
           question_type, or any evaluator output could be passed, by
           accident or otherwise. Generator itself carries no gold-related
           attribute (see heinzy/generation/generator.py), so passing it for
           its chat plumbing cannot leak gold data either.
invariant: exactly one additional LLM call. Reuses Generator._chat() for the
           HTTP/auth/provider plumbing -- never Generator.generate(), which
           would produce a second first-pass draft, forbidden by the
           experiment spec -- and always with tools=None, so this call is
           tool-free regardless of Generator.use_tools.
invariant: malformed verifier output never becomes an invented answer. Any
           JSON parse failure, missing/invalid decision, or missing
           final_text on REVISE falls back to a conservative ABSTAIN using
           the generator's own configured refusal_text, with the raw output
           and a parse_error preserved for audit.
invariant: refused drafts are verified exactly like answered drafts. This
           module never special-cases draft.refused -- a Layer 1 (no
           context) or Layer 2 (sentinel) refusal is passed through
           unconditionally so the verifier can catch a false refusal.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from heinzy.generation.generator import Answer, Generator, extract_usage
from heinzy.retrieval.store import ScoredChunk

DECISION_KEEP = "KEEP"
DECISION_REVISE = "REVISE"
DECISION_ABSTAIN = "ABSTAIN"
_VALID_DECISIONS = {DECISION_KEEP, DECISION_REVISE, DECISION_ABSTAIN}

# Bump this (and expect the fingerprint below to change) any time the prompt
# text changes, so a report's verifier_prompt_version/fingerprint pair always
# identifies exactly which instructions produced its verifier calls.
#
# v1 -> v2: developed from human-reviewed failure analysis of the 30-question
# pilot (see the pilot's annotation observations). v2 adds explicit checks for
# completeness (an answer can be all-true and still omit a material part),
# scenario-specific reasoning (apply the user's stated conditions to a
# retrieved rule/range instead of repeating it generically), false-refusal
# recovery, a stricter definition of evidence sufficiency (resolves every
# material part, not just "a related chunk exists"), unsupported-exclusivity
# inference, and citation-label discipline (cite the actual section heading,
# never a body sentence). These are general reasoning instructions, not
# tuned to any specific pilot question, ID, or fact -- see verify_answer()'s
# docstring invariant on gold isolation, which v2 does not touch.
VERIFIER_PROMPT_VERSION = "v2"

# Deliberately narrow inputs: only the question, the retrieved excerpts, and
# the first-pass draft ever get formatted into this. There is no gold data
# anywhere in scope when this string is built (see _build_user_prompt).
_VERIFIER_SYSTEM_PROMPT = (
    "You are the second pass of a two-pass answering system for Heinz "
    "College MISM advisors. The first pass already produced a draft answer. "
    "Your job is to check that draft against the SAME retrieved handbook "
    "excerpts it was given, and decide whether to keep it, revise it, or "
    "abstain.\n\n"
    "RULES:\n"
    "1. Use ONLY the retrieved excerpts below. Never use outside knowledge, "
    "general knowledge about universities, or anything you know about "
    "Carnegie Mellon that is not in the excerpts. Never infer a handbook "
    "fact that the excerpts do not state.\n\n"
    "2. First, work out what the question actually requires. Identify its "
    "material parts: every condition, comparison, constraint, or specific "
    "value the user is asking for. A response that is merely related to the "
    "topic, or that answers a nearby question instead of the one actually "
    "asked, is not sufficient.\n\n"
    "3. Check completeness. Does the draft resolve every material part you "
    "identified, using the excerpts? If the excerpts contain information "
    "that would fill a part the draft left out, omitted, or answered only "
    "vaguely (for example, noting that a limit or value exists without "
    "stating it), that is an incomplete answer. Being factually correct in "
    "what it does say is not enough if it leaves a material part "
    "unresolved.\n\n"
    "4. Check scenario-specific reasoning. If the user described a specific "
    "situation (a number, a status, a choice between options) and the "
    "excerpts state a general rule or range, work out what that rule "
    "implies for the user's specific situation. Prefer the most specific "
    "conclusion the excerpts actually support over repeating a generic rule "
    "or range that could have been narrowed down. Do this only from what "
    "the excerpts state -- do not use outside knowledge to bridge the gap.\n\n"
    "5. Check refusals too. A draft that declines to answer, abstains, or "
    "says the information isn't available is not automatically correct. If "
    "the excerpts actually contain enough evidence to resolve the "
    "question's material parts, that is a false refusal -- revise it into "
    "a substantive, supported answer in this same reply.\n\n"
    "6. Judge evidence sufficiency correctly. Sufficient evidence means the "
    "excerpts resolve ALL of the question's material parts, not merely that "
    "one related excerpt was retrieved. A fact that is topically related "
    "but does not actually answer a required part does not count. If even "
    "one material part has no supporting evidence in the excerpts, do not "
    "guess, generalize, or present a partial answer as if it were "
    "complete.\n\n"
    "7. Never make an inference the excerpts do not support. In particular: "
    "a statement that one group or case is permitted something does NOT by "
    "itself mean every other group or case is prohibited from it -- only "
    "conclude exclusivity or prohibition if the excerpts actually say so. "
    "More definite-sounding wording is not automatically more accurate.\n\n"
    "8. Citations. Every material factual claim in your final_text must be "
    "supported by one of the excerpts below, cited in the form "
    '(see "<exact section heading copied from the excerpts>"). Cite the '
    "actual section heading as it appears in the excerpts -- never a body "
    "sentence, a paraphrase, or a section that is not among the excerpts. A "
    "citation only counts if the section it names actually supports the "
    "claim it is attached to; a citation that exists but does not back its "
    "claim is as much a problem as no citation at all.\n\n"
    "9. Decide exactly one of:\n"
    "   KEEP    - the draft already directly resolves every material part "
    "of the question, is correct and complete relative to the excerpts, "
    "correctly applies the user's specific scenario, and is properly "
    "cited. (This includes correctly abstaining when the excerpts truly do "
    "not resolve the question's material parts.)\n"
    "   REVISE  - the excerpts are sufficient to resolve the question's "
    "material parts, but the draft has a fixable problem: incomplete "
    "coverage, an omitted condition, an unnecessarily generic answer where "
    "a specific one is supported, a false refusal, or a repairable "
    "citation. Write the corrected, complete final answer yourself, right "
    "now, in this same reply -- there will not be another chance.\n"
    "   ABSTAIN - the excerpts do not provide enough evidence to resolve "
    "one or more material parts of the question. Say plainly that the "
    "CURRENT RETRIEVED CONTEXT does not provide enough evidence -- never "
    "claim that the handbook itself lacks the information, since you only "
    "see what was retrieved for this question.\n\n"
    "10. Reply with ONLY a single JSON object, no prose before or after "
    "it, and no markdown code fence around it. Match this shape exactly:\n"
    '{"decision": "KEEP|REVISE|ABSTAIN", "final_text": "...", "reason": "..."}\n'
    "final_text is the complete answer text the advisor should see (for "
    "KEEP, restate the draft's own answer; for REVISE, the corrected and "
    "now-complete answer; for ABSTAIN, a brief note is fine, it will be "
    "replaced with the standard refusal text). reason is a short, "
    "one-or-two-sentence explanation of your decision, for an auditor, not "
    "the advisor."
)


def verifier_prompt_fingerprint() -> str:
    """Stable hash of the verifier system prompt, for provenance/rerun-matching."""
    return hashlib.sha256(_VERIFIER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class VerificationResult:
    decision: str
    final_text: str
    refused: bool
    reason: str
    raw_verifier_output: str
    model_tag: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    # Always True from verify_answer(): the _chat call is made unconditionally
    # before any parsing, on every return path including the ABSTAIN
    # fallback, so a verifier result never represents "no call was made" --
    # only "the call happened but its usage/output couldn't be used/trusted".
    model_call_made: bool
    # Set only when the verifier's own output could not be trusted and this
    # result is the conservative ABSTAIN fallback -- see module invariant.
    parse_error: str | None = None


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verifier_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort JSON extraction. Returns (parsed, error); exactly one is None."""
    text = (raw or "").strip()
    if not text:
        return None, "empty verifier response"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    # Models sometimes wrap JSON in prose or a markdown code fence despite
    # being told not to; grab the first {...} block and try again.
    match = _JSON_BLOCK.search(text)
    if not match:
        return None, "no JSON object found in verifier response"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as exc:
        return None, f"JSON block found but did not parse: {exc}"


def _build_user_prompt(question: str, draft: Answer, hits: list[ScoredChunk]) -> str:
    context = "\n\n".join(
        f"[{h.section_path}] (p{h.source_pages}): {h.text}" for h in hits
    )
    draft_status = "declined to answer" if draft.refused else "answered"
    return (
        f"Retrieved handbook excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"First-pass draft ({draft_status}):\n{draft.text}"
    )


def _fallback_abstain(
    raw: str,
    reason: str,
    refusal_text: str,
    model_tag: str | None,
    usage: dict[str, int | None],
    latency: float,
) -> VerificationResult:
    return VerificationResult(
        decision=DECISION_ABSTAIN,
        final_text=refusal_text,
        refused=True,
        reason=f"verifier output could not be used safely: {reason}",
        raw_verifier_output=raw,
        model_tag=model_tag,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_seconds=latency,
        model_call_made=True,
        parse_error=reason,
    )


def verify_answer(
    question: str,
    draft: Answer,
    hits: list[ScoredChunk],
    generator: Generator,
) -> VerificationResult:
    """Run the single Treatment verification pass over an already-produced draft.

    generator supplies mechanics only (chat endpoint/auth/model/sampling and
    the configured refusal_text fallback) via its `_chat` method -- the same
    HTTP/provider plumbing Generator.generate() itself uses, reused here
    instead of duplicated. It is never asked to generate() again, and it
    carries no gold/Row attribute, so there is nothing on this call site
    that could leak gold data even by mistake.
    """
    messages = [
        {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, draft, hits)},
    ]

    t0 = time.monotonic()
    data = generator._chat(messages, tools=None)
    latency = time.monotonic() - t0

    raw = (data.get("message") or {}).get("content") or ""
    usage = extract_usage(generator.provider, data)
    model_tag = generator.model_tag

    parsed, error = _parse_verifier_json(raw)
    if parsed is None:
        return _fallback_abstain(raw, error, generator.refusal_text, model_tag, usage, latency)

    decision = parsed.get("decision")
    if not isinstance(decision, str) or decision.strip().upper() not in _VALID_DECISIONS:
        return _fallback_abstain(
            raw, f"missing or invalid decision: {decision!r}",
            generator.refusal_text, model_tag, usage, latency,
        )
    decision = decision.strip().upper()

    reason = parsed.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else ""

    if decision == DECISION_KEEP:
        return VerificationResult(
            decision=DECISION_KEEP,
            # Exact draft text, per spec -- never trust the verifier to have
            # copied it verbatim, since paraphrase-on-"repeat" is a real
            # model failure mode.
            final_text=draft.text,
            refused=draft.refused,
            reason=reason,
            raw_verifier_output=raw,
            model_tag=model_tag,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=latency,
            model_call_made=True,
        )

    if decision == DECISION_REVISE:
        final_text = parsed.get("final_text")
        if not isinstance(final_text, str) or not final_text.strip():
            return _fallback_abstain(
                raw, "decision was REVISE but final_text was missing/empty",
                generator.refusal_text, model_tag, usage, latency,
            )
        return VerificationResult(
            decision=DECISION_REVISE,
            final_text=final_text,
            refused=False,
            reason=reason,
            raw_verifier_output=raw,
            model_tag=model_tag,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=latency,
            model_call_made=True,
        )

    # DECISION_ABSTAIN: standardized refusal text, never the verifier's own
    # prose, so downstream code can rely on refused=True + a deterministic
    # string exactly like Layer 1/2 refusals already do.
    return VerificationResult(
        decision=DECISION_ABSTAIN,
        final_text=generator.refusal_text,
        refused=True,
        reason=reason,
        raw_verifier_output=raw,
        model_tag=model_tag,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_seconds=latency,
        model_call_made=True,
    )
