# Heinzy RAG Evaluation — LLM-as-a-Judge Rubric v1.2 (citation + semantic calibrated candidate)

**Status: DEVELOPMENT-SET CALIBRATION CANDIDATE, NOT YET RUN, NOT
VALIDATED.** This is v1.1 (`judge_rubric_v1.1.md`, citation calibration
only) **plus** a small set of semantic clarifications, made in response to
specific manually-adjudicated judge/human disagreements found in the first
real development-set Judge run (GPT-OSS 20B, 30 questions × 2 arms;
pre-calibration semantic-axis agreement 251/270 = 93.0% — see
`judge_calibration_log_v1.2.md` for the full record). **Citation
resolution, citation hierarchy, citation normalization, and citation-scope
behavior are carried forward from v1.1 completely unchanged in this
document** (§6's hierarchy-aware evidence-scope subsection and all of §7
are byte-for-byte identical to v1.1, which is itself byte-for-byte
identical to v1 in those same sections) — only the semantic axes
(Correctness, Completeness, Faithfulness, plus one Citation Support
addendum that does not touch resolution/scope rules) were clarified beyond
v1.1. This candidate has not itself been run against any response yet, and
post-calibration agreement on this same 30-question development set — once
it is run — will **not** constitute independent validation; see
`judge_calibration_log_v1.2.md` and `judge_calibration_plan_v1.md` §1/§5
for why.

This document is the human-readable rubric backing `judge_prompt_v1.2.txt`
and `judge_output_schema_v1.json` (the output schema is unchanged from v1).
It exists so a reviewer can inspect the grading logic without reading the
raw prompt, and so future prompt revisions have a stable reference to diff
against.

---

## 0. Methodology status (read this before anything else)

The current 30-question / 60-response pilot (`eval/questions_pilot_v1.1.jsonl`,
`eval/annotation/final_human_labels_v2.1.csv`) is **judge-development data
only**. No subset of it is, or may later be claimed to be, an independent
judge validation set — see
`eval/annotation/judge_split_feasibility_audit_v1.md` for the full audit.
Concretely, this development set has:

- **zero** Faithfulness FAIL examples,
- **zero** Correct Abstention / Unsupported Answer FAIL examples,
- exactly **one** Correctness FAIL example, and it belongs to a
  rubric-development-exposed family,
- exactly **one** independent (non-exposed) Citation Validity/Citation
  Support FAIL example, and the two axes have never failed independently of
  each other in this data.

Pilot agreement numbers are useful for **debugging the rubric and prompt**.
They must never be reported as evidence of failure-detection reliability,
and optimizing this prompt to raise aggregate agreement on this PASS-heavy
set is an explicit anti-goal. A frozen judge requires **new, independent
question families** (never generated yet) before its reliability can be
claimed — see `judge_calibration_plan_v1.md` §4.

---

## 1. Axis inventory

| # | Axis | Kind | Applies to |
|---|---|---|---|
| 1 | Task Completion | Semantic (LLM) | Answerable |
| 2 | Correctness | Semantic (LLM) | Answerable |
| 3 | Completeness | Semantic (LLM) | Answerable |
| 4 | Faithfulness | Semantic (LLM) | Answerable |
| 5 | Citation Support | Semantic (LLM) | Answerable |
| 6 | Citation Validity | **Deterministic** (non-LLM) | Answerable |
| 7 | Correct Abstention | Semantic (LLM) | Unanswerable |
| 8 | Unsupported Answer | Semantic (LLM) | Unanswerable |
| — | Final GAS | Deterministic composition, downstream of 1–6 | Answerable only |

The LLM judge is **never** asked to produce Citation Validity or Final GAS.
Both are computed outside the LLM call (§7 for Citation Validity, §10 for Final GAS).

---

## 2. Task Completion

**Definition.** Whether the response substantively attempts to resolve the
user's actual requested task. This is a semantic axis — never define it as
`refused == false`, non-empty text, or a length threshold. Standardized pure
refusals may be mechanically detectable, but the boundary between "a real,
if partial, attempt" and "a non-attempt dressed as an attempt" requires
judgment.

- **PASS** — the response substantively attempts to resolve the user's
  actual question, even if the content turns out incorrect or incomplete.
- **FAIL** — pure refusal; an answer to a different/unrelated question;
  merely repeating or rephrasing the question back; or otherwise materially
  failing to attempt the requested task.

**Critical separation:** a response that substantively attempts only *part*
of a multi-part question is **not** automatically Task Completion FAIL. Task
Completion asks "did it try to engage with what was actually asked", not
"did it fully succeed." A response can be Task Completion PASS and
Completeness FAIL in the same evaluation — these must never be conflated.

There is no N/A or REVIEW_REQUIRED-by-default here; the enum is `PASS |
FAIL | REVIEW_REQUIRED`, and REVIEW_REQUIRED is reserved for genuinely
ambiguous cases (e.g., a response that engages tangentially with the topic
in a way that could plausibly be read either as a real attempt or as an
evasion).

**Do not overuse REVIEW_REQUIRED.** Most responses are not boundary cases:
a substantive attempt at the actual question, including a substantive
partial answer to a multi-part question, is PASS; a pure refusal, an answer
to a different question, a mere restatement of the question, or another
materially-no-attempt response is FAIL. REVIEW_REQUIRED is for the narrow
remainder where it is genuinely unclear whether the response is a real (if
weak) attempt versus tangential engagement or evasion — not a default
fallback whenever a response is short, hedged, or imperfect.

---

## 3. Correctness

**Inputs used:** `question`, `generated_response`, `gold_answer`,
`gold_claims`, `supporting_evidence` — the **primary** correctness
references, checked first for every material claim. `retrieved_context` is
available as a **fallback verifier**, used only per the evidence policy
below, for a material claim gold does not address at all. **Never** outside
knowledge, from either reference.

- **PASS** — every material factual claim in the response is *verifiable*
  from the supplied correctness references (gold, and — only when gold is
  silent on that specific claim — `retrieved_context`; see evidence policy
  below), **and** every such claim is correct. Both conditions are
  required: PASS is not available merely because no claim was
  *contradicted* — an unverifiable claim blocks PASS just as a contradicted
  one does (see REVIEW_REQUIRED below; do not resolve an unverifiable claim
  to PASS or to FAIL).
- **FAIL** — at least one material factual claim is contradicted by, is
  inconsistent with, or materially misapplies the supplied correctness
  references (including applying a correct general rule incorrectly to a
  specific stated scenario).
- **N/A** — no material factual claim is made at all (e.g., a pure refusal).
  Do not use N/A for a claim that exists but cannot be verified — that is
  REVIEW_REQUIRED, not N/A; these two states must remain distinct.
- **REVIEW_REQUIRED** — a material claim is made, but neither gold nor (when
  checked, per the evidence policy below) `retrieved_context` verifies or
  contradicts it. This is the **only** correct disposition for a claim that
  remains unverifiable after both checks: do not PASS it, and do not FAIL
  it merely because it is absent from gold. List every such claim in
  `unverifiable_claims`.

**The two-sided guardrail (memorize this, it is the most common failure
mode for this axis):** absence from gold does **not** automatically mean a
claim is false, and inability to contradict a claim does **not**
automatically mean it is correct. When a material claim cannot be judged
from the supplied references, the answer is REVIEW_REQUIRED — never a
guess in either direction.

**Correctness evidence policy — gold is primary; retrieved context is a
fallback verifier for claims gold does not cover (v1.2 clarification).**
This closes a specific gap: previously, "not in gold" alone could push a
true, retrieval-supported claim straight to REVIEW_REQUIRED even though the
same authoritative material supplied to the judge for this response already
verified it.

1. Check every material claim against gold first — gold remains the primary
   reference and is checked before anything else.
2. If a material claim is not addressed by gold at all (gold neither
   confirms nor contradicts it), the judge **may** additionally check that
   specific claim against the supplied `retrieved_context`.
3. A claim verified by *either* source — gold, or `retrieved_context` when
   gold was silent — may be judged correct. A claim contradicted by either
   source is still FAIL.
4. If **neither** source verifies or contradicts the claim, REVIEW_REQUIRED
   remains the correct disposition — this policy expands what counts as
   *verification*, it does not remove REVIEW_REQUIRED for claims that are
   genuinely unverifiable from everything supplied.
5. Never use outside/world knowledge at either step.

**This does not collapse Correctness into Faithfulness — they stay
independent axes asking different questions.** Correctness asks whether a
claim is *true* according to the available authoritative reference material
(gold, falling back to `retrieved_context` only when gold is silent on that
claim). Faithfulness (§5) asks whether the claim is *grounded in the
retrieved context actually supplied to this specific response*, regardless
of what gold says. A claim can therefore be **Correctness PASS while
Faithfulness FAILs** — true according to gold/reference evidence, but this
particular response's own retrieved context doesn't happen to support it —
and that divergence is expected, not a contradiction to resolve away.

**Deterministic derivation counts as verification.** A claim that follows
directly and deterministically — by simple arithmetic or logical necessity,
with every needed premise present in the correctness references and no
additional unstated assumption — from stated facts may be judged correct
even when the exact conclusion is not written verbatim anywhere in the
references. A merely *possible* inference, or a conclusion that requires
choosing among multiple facts/allocations compatible with what was stated,
is **not** sufficient. See §5's deterministic-derivation principle and
quantity-identity rule, which apply identically here.

**Explicitly out of scope for this axis:** missing required information is
a Completeness concern, not a Correctness concern. A response that says
less than required, but says nothing wrong, is Correctness PASS.

---

## 4. Completeness

**Definition.** Whether the response conveys the frozen `gold_claims` for
this specific question — the question-level required content, as finalized
by human review, not a fresh gold interpretation performed by the judge.

- **PASS** — every required gold claim needed for the frozen
  minimally-complete answer is meaningfully and correctly conveyed.
  Semantically equivalent wording is acceptable; exact wording is never
  required. A single sentence can jointly convey more than one gold claim.
- **FAIL** — one or more required gold claims are missing, or a required
  claim is present but conveyed incorrectly.
- **REVIEW_REQUIRED** — the frozen gold/rubric appears internally
  inconsistent for this question; a listed gold claim appears clearly
  irrelevant to the actual question asked; or a literal application of the
  frozen gold would conflict with established rubric principles.

**Hard constraint — read this twice:** the judge must never redesign gold
requirements on a question-by-question basis, and must never silently treat
a required gold claim as non-required because it seems unnecessary in
context. If the judge suspects a gold-design problem, it must: (1) preserve
the frozen requirement as-is for grading purposes, (2) set
`rubric_or_gold_conflict: true`, (3) return `REVIEW_REQUIRED`, and (4) leave
the resolution to human adjudication. **See §11 for a live, real example of
exactly this situation found during rubric design** — the judge's inability
to autonomously reach the same conclusion a human reviewer reached there is
intentional, not a defect.

**General lessons carried over from human review** (stated abstractly, not
tied to specific pilot content):
- Evaluate the *minimum* content needed to resolve the question actually
  asked — not everything gold happens to know about the topic.
- Do not reward verbosity. Do not penalize a concise answer that is
  nonetheless sufficient.
- A single joint statement can satisfy more than one gold claim at once;
  don't require a 1:1 sentence-to-claim mapping.
- Watch for responses that convey a *related* fact (e.g., a bound, limit, or
  category membership) that only *jointly implies* a required claim rather
  than stating it directly — deciding whether that joint implication is
  "meaningfully conveyed" versus "not actually conveyed" is a genuinely hard
  judgment call and a likely source of judge/human disagreement worth
  tracking during calibration.

**Material numeric details cannot be omitted (v1.2 clarification).** When a
required gold claim's substance is a specific numeric value (a unit count, a
GPA, a deadline, a cap), a response that only gestures at the *existence* of
that requirement — without stating the value — has **not** conveyed the
claim, even if it uses confident, complete-sounding language. "There is a
minimum unit requirement" is not semantically equivalent to "the minimum is
162 units," and does not satisfy a gold claim whose substance is the number
162. This is not a verbatim-wording requirement (§4's general rule that
semantically equivalent phrasing is fine still applies) — it is that the
material *value itself*, not just the *category* of requirement, is part of
what "meaningfully conveyed" means whenever gold's claim specifies one.

**Necessary entailment may satisfy a gold claim without verbatim statement
(v1.2 clarification, generalizing a distinction human review already
applied).** A gold claim may be treated as conveyed if it follows as a
**logically necessary** consequence of what the response actually states,
with no additional unstated assumption — not merely as one *plausible*
reading among several. Three examples that must be told apart:

- **A. Not necessary — a possible inference is insufficient.** A response
  states a total (e.g. "162 units over three semesters") with no other
  constraint. This does **not** entail an equal per-period split (e.g. "54
  units each semester") — 162 could equally be split unevenly (e.g.
  100 + 30 + 32). Do not treat an arithmetic average (162 ÷ 3 = 54) as if it
  were a stated or necessarily-true value: computing an average is not the
  same as the response having conveyed that value. If the required gold
  claim is the *per-period* figure, this response leaves it FAIL.
- **B. Necessary — an additional stated constraint forces a unique
  conclusion.** A response states the same total AND an explicit per-period
  upper bound that, combined with the total and period count, allows only
  one possible allocation (e.g. "162 units over three semesters" **and**
  "no semester may exceed 54 units" — three periods, each capped at 54,
  summing to 162, forces every period to equal exactly 54; no other
  non-negative allocation satisfies both stated constraints
  simultaneously). Here the per-period figure **is** necessarily entailed,
  even though the response never states "54 units per semester" directly,
  and Completeness may PASS for that required claim.
- **C. Still not necessary — multiple allocations remain possible.** If the
  additional stated constraint does not narrow the possibilities to exactly
  one (e.g. a *range* rather than a tight cap, or a cap that still permits
  more than one valid allocation), the conclusion is not necessarily
  entailed and the claim is not satisfied merely by that partial
  narrowing — treat it the same as case A.

The general principle: **a possible inference is insufficient; a simple
arithmetic average is insufficient; a conclusion that requires choosing
among multiple allocations compatible with what was stated is
insufficient.** Only a genuinely unique, forced consequence of the
response's own stated facts — with no assumption imported from outside the
response — satisfies a gold claim this way.

---

## 5. Faithfulness

**Inputs used:** `retrieved_context` **only**. Never `gold_answer`, never
`supporting_evidence`, never outside/handbook knowledge of any kind — even
if the judge happens to "know" the retrieved context is incomplete or that
gold would say something different. Faithfulness asks one question only:
*given what was actually retrieved for this response, is every material
claim in the response supported by it?*

- **PASS** — every material factual claim in the response is supported by
  the supplied retrieved context.
- **FAIL** — at least one material factual claim is unsupported by the
  retrieved context, contradicts it, materially extrapolates beyond what it
  establishes, or draws an unsupported exclusivity/prohibition/universality/
  generalization inference from it (see principle below).
- **N/A** — no material factual claim is made.
- **REVIEW_REQUIRED** — the relationship between a material claim and the
  retrieved evidence is genuinely ambiguous and cannot be reliably resolved
  from the supplied context alone.

**Generalization-inference principle (apply this literally, do not weaken
it):** evidence that "X is permitted/true for group or case A" does **not**
by itself establish "only A" or "not for any other group or case" — that
additional exclusivity claim requires its own explicit textual support.
This applies symmetrically to any unsupported universal, exclusive, or
prohibitive claim manufactured from a merely-positive statement.

**Deterministic derivation and numeric-quantity identity (v1.2
clarification — this principle is shared by Correctness §3 and Citation
Support §6, stated once here in full).**

**A. Deterministic derivation is grounded.** A factual conclusion may be
Faithfulness PASS even when the exact conclusion is not written verbatim in
`retrieved_context`, provided it follows directly and deterministically —
by simple arithmetic or logical necessity — from facts that *are* stated
there. Example: if the retrieved context states a concentration requires 36
total units, two fixed 6-unit courses, and a third course offered as either
6 units (option A) or 12 units (option B), then "choosing option A leaves
36 − 6 − 6 − 6 = 18 additional units to complete" is a deterministic
arithmetic consequence of exactly those stated numbers — it may be judged
grounded even though "18" itself never appears in the retrieved text.

**B. Every premise must be present in the relevant evidence.** All material
numbers/facts the derivation depends on must themselves be supported by the
retrieved context (for Faithfulness) or by the citation's own allowed
evidence scope (for Citation Support, §6) — a derivation is only as grounded
as its weakest premise. If even one number the arithmetic depends on is not
actually present in the relevant evidence, the derived conclusion is not
grounded there.

**C. No extra assumptions, no outside knowledge.** Do not allow open-ended
reasoning, real-world knowledge, or any assumption not present in the
supplied evidence to fill a gap in the derivation — this is the same
no-unstated-assumption principle as Completeness §4's necessary-entailment
rule, applied in the opposite direction (there: does the response's own
content entail a required gold claim; here: is the response's own derived
claim entailed by the retrieved evidence).

**D. Numeric-quantity identity — do not call two different numbers
"contradictory" without first checking what each one measures.** Two
numeric statements in a response are not automatically inconsistent merely
because the numbers differ. Determine whether they refer to the **same
quantity** before concluding a contradiction. Continuing the concentration
example: "18 additional units are needed under the option-A path" and "the
two course options differ from each other by 6 units" are **two different
quantities** — one is *units still needed to reach the 36-unit total after
choosing option A*, the other is *the unit gap between option A and option
B themselves* (option B is 12, option A is 6, difference = 6). Both can be
true and non-contradictory at once; treating "18 ≠ 6" as a contradiction
without first identifying what each number measures is exactly the error
this rule exists to prevent.

---

## 6. Citation Support

**Inputs used:** the response's material claims, its attached
citations/`cited_sections`, and the specific retrieved text each citation
resolves to. This axis is evaluated **after** citation resolution — it asks
whether the *evidence a citation points to* actually backs the claim it's
attached to, which is a different question from whether the citation label
itself is valid (that's §7).

- **PASS** — every material factual claim requiring evidence has an
  appropriate citation, and the text that citation resolves to substantively
  supports that specific claim.
- **FAIL** — a material claim lacks any citation; a cited passage does not
  support the claim it's attached to; a citation supports only part of a
  broader material claim while the rest goes uncited/unsupported; or a
  citation is attached to the wrong claim (misattribution).
- **N/A** — no material factual claim exists that would require a citation.
- **REVIEW_REQUIRED** — claim-to-citation support cannot be reliably
  determined from the supplied structured citation/retrieval information.

**Deterministic derivation within the citation's allowed evidence scope
(v1.2 clarification).** A cited claim that is a deterministic arithmetic or
logical consequence of facts present in that citation's already-computed
allowed evidence scope (§7's hierarchy-aware scope — cited section plus its
retrieved descendants) may be Citation Support PASS, using the same
deterministic-derivation and numeric-quantity-identity principles as §5
Faithfulness (parts A–D there apply here identically, substituting "the
citation's allowed evidence scope" for "retrieved_context"). Every premise
the derivation depends on must itself be present within that scope — a
derivation resting on a fact from outside the cited section's allowed
scope is not citation-supported, even if that fact exists elsewhere in
`retrieved_context` (that is a Faithfulness question, not a Citation
Support one — see Case C below). **This clarification does not change
resolution or scope itself in any way** — it only clarifies how *support*
is judged once a citation has already resolved and its scope has already
been computed by the deterministic resolver; citation resolution,
hierarchy, normalization, and scope computation are unchanged from v1.

**Interaction rule with Citation Validity and Faithfulness — three explicit
cases.** Citation Validity, Citation Support, and Faithfulness are graded
independently (§0/§13), but "independent axes" does **not** mean the same
underlying defect can never cause multiple axes to fail. A single bad
citation can legitimately fail more than one axis at once; the cases below
tell you exactly when that is expected, not a contradiction to resolve away.

- **Case A — citation points to a nonexistent source/section, or a
  section not present in the retrieved context supplied to this response.**
  The deterministic Citation Validity evaluator (§7) resolves this to
  **FAIL**. By definition there is no real cited evidence for the semantic
  judge to check support against. Citation Support for the material claim
  relying on that citation must also be evaluated as **FAIL** (not N/A) —
  there is no usable cited retrieved evidence supporting the claim, so the
  claim is effectively uncited for support purposes. (Citation Validity
  itself is a downstream composition input, not computed by the semantic
  judge — see §7.)
- **Case B — citation formatting differs superficially but resolves
  unambiguously under the predefined exact canonical normalization rules
  (§7) to a section that actually was retrieved.** Citation Validity **may
  PASS** here (exact resolution succeeded after normalization — case-fold,
  whitespace/punctuation collapsing, per §7; this is normalization, not
  fuzzy semantic matching, and no fuzzy semantic matching is introduced by
  this rule). Citation Support is then judged **independently**, purely on
  whether the resolved evidence text actually supports the claim it is
  attached to — a validly-resolving citation can still fail Citation
  Support if the passage it resolves to does not substantively back the
  claim.
- **Case C — the material claim is supported somewhere in
  `retrieved_context`, but the citation attached to that specific claim
  points elsewhere, or points to something that does not support it.**
  Faithfulness **may PASS** (the claim itself is grounded somewhere in what
  was retrieved), while Citation Support **FAILs** (misattribution, or a
  citation that does not back the claim it decorates) — Faithfulness asks
  whether the claim is grounded in what was retrieved at all; Citation
  Support asks whether *this response's own citation* actually backs *this
  claim*. These are genuinely different questions and are expected to
  diverge on a misattributed-citation response.

**Hierarchy-aware evidence scope (adopted policy).** A citation's evidence
scope is **not** limited to the literal retrieved chunk it happens to sit
inside. The rule:

> A citation to a section covers evidence in that section and its
> descendants within the retrieved context. A citation to a subsection
> covers only that subsection and its descendants — not its parent, and not
> its siblings. Coverage is one-directional: parent → descendants only,
> never descendant → ancestor, never sibling → sibling.

Concretely (see §7 for how the deterministic resolver computes this):

- A citation to section `8` may be supported by evidence anywhere in `8`,
  `8.1`, `8.2`, `8.2.1`, `8.2.2`, etc. — including a subsection that was
  retrieved as its **own separate chunk**, not only text embedded in the
  same chunk as `8` itself. Do not fail Citation Support merely because the
  supporting text lives in a descendant subsection rather than verbatim
  under the cited section's own literal chunk text.
- A citation to `8.2.2` may be supported only by evidence in `8.2.2` or a
  descendant of `8.2.2`. Evidence that exists only in `8.2.1` (a sibling),
  `8.2` (the parent), or `8` (the grandparent) does **not** satisfy a
  citation to `8.2.2` — treat that as **FAIL** (a citation attached to the
  wrong, too-narrow or unrelated, section for what it's actually backing),
  not as a resolution ambiguity.
- **Broader citation specificity is not itself a Citation Support
  failure.** Citing a parent section when a subsection would have been more
  precise is not penalized on this axis — this project does not currently
  score citation precision/specificity as a separate metric. Only fail
  Citation Support when the cited section's allowed scope (itself +
  descendants) genuinely does not contain supporting evidence for the
  claim, never merely because a more specific citation was available.
- A subsection heading embedded **inside** a retrieved parent chunk's text
  (e.g. `8.2.1. Requirements` appearing as a heading line inside the chunk
  retrieved for `8`) is a valid, independently citable logical section —
  the response does not need a separate retrieval chunk per subsection for
  a citation to that subsection to resolve. See §7.

---

## 7. Citation Validity — deterministic evaluator contract

Citation Validity is **not** an LLM judgment. It is defined here as an
exact, auditable, deterministic contract so that (a) it never varies run to
run, and (b) its failure modes are documented and testable independently of
any prompt. **This task defines the contract only; it is not implemented
here.**

**States the evaluator must distinguish:**

| State | Meaning |
|---|---|
| A. No citation present | The response makes no citation at all for a given claim/section reference. |
| B. Citation present and valid | The citation resolves, under an exact defined format, to a section/source identifier actually present in the retrieved context supplied to that response. |
| C. Citation present but invalid | The citation does not resolve under (B). |

- **PASS** — every citation used in the response resolves exactly to a
  retrieved source/section identifier, under the project's defined citation
  format.
- **FAIL** — at least one citation references a nonexistent section/source,
  references a section/source that was not part of the retrieved context
  actually supplied to this response, or otherwise fails exact resolution
  under the defined format.
- **N/A** — no citation is present in the response at all.

**Exact normalization + lookup contract (recommended design, not
implemented here):** a citation string should be normalized (case-folding,
whitespace/punctuation collapsing) and then looked up against the *exact
set of section identifiers present in the retrieved context supplied to
that specific response* — never against the full handbook's section list,
since a citation to a real handbook section that was not actually retrieved
for this response is exactly as invalid as a citation to a nonexistent
section (the response could not legitimately have grounded a claim in text
it was never given).

**Documented failure risks a deterministic evaluator must explicitly guard
against** (do not silently accept any of these as "close enough"):
- **Fuzzy substring matching** — a naive "does this text appear somewhere in
  a retrieved chunk" check will pass citations that are not, in fact, valid
  section identifiers; substring presence is not the same as identifier
  resolution.
- **Numeric sibling collisions** — sibling or nested numeric section
  identifiers that differ only by suffix (e.g. `N.M` vs `N.M.K`) can be
  conflated by loose numeric-prefix matching; the evaluator must match
  identifiers exactly, not by numeric-prefix containment.
- **Numeric/page-like parsing ambiguity** — page numbers, unit counts, or
  other bare numerals appearing near a citation must not be mistaken for a
  section identifier.
- **Body-text-as-citation** — a response may quote or paraphrase a sentence
  from retrieved body text and present it as if it were the citation label,
  rather than citing the actual section heading/identifier. This must
  resolve to FAIL (or N/A if no real citation attempt exists), never PASS,
  even though the quoted text is verbatim-present in the retrieved context.

**Hierarchical section ambiguity — RESOLVED, adopted policy (formerly an
open risk flagged for "an explicit, documented resolution rule, not an
implicit close-enough heuristic").** That rule is now defined:

1. **Logical section index, not flat chunk list.** The evaluator builds a
   deterministic index of every logical section reachable from what was
   actually retrieved for this response: each retrieved chunk's own
   top-level section, PLUS every subsection heading detected embedded
   inside that chunk's text (e.g. a chunk retrieved under section `8` whose
   body contains heading lines for `8.1`, `8.2`, `8.2.1`, `8.2.2`). A
   subsection does not need its own separate retrieval chunk to be a valid,
   independently citable logical section — see §2's embedded-heading rule.
   Heading detection is line-based and length-bounded (never fuzzy):
   a candidate heading must occupy its own line with a short, bounded
   title; a heading whose title runs on into body prose on the same source
   line is not indexed as a spurious node (its text simply remains inside
   its actual parent's own content, never lost).
2. **Distinct nodes, one-directional coverage.** `8`, `8.2`, and `8.2.1` are
   three distinct nodes, identified by their parsed dotted section number.
   A citation resolves to **exactly one** logical section node — never
   "the section and everything nearby." What that resolved node's evidence
   is allowed to include for **Citation Support** purposes (its descendants,
   never its ancestors or siblings) is the hierarchy-scope rule in §6, not
   a Citation Validity concept — Validity is only "does this citation
   identify one real retrieved logical section," never "how much may it
   cover."
3. **Number-omitted title normalization — narrow, exact, and unique-only.**
   A citation that omits the section number but exactly matches (after the
   same canonical case/whitespace/quote normalization used everywhere else
   in this contract) the bare TITLE of exactly one logical section in the
   index resolves to that section (e.g. citing `"Internship Requirement"`
   when the only matching retrieved logical section is `9. Internship
   Requirement`). This is still exact-equality matching on a normalized
   string — never a fuzzy/partial/stemmed comparison. If the number-omitted
   title matches more than one distinct logical section, the result is
   **AMBIGUOUS**, never guessed; if it matches none, **UNRESOLVED**.
4. **Still no fuzzy/semantic matching anywhere in this contract.** Not for
   heading detection (line-based, length-bounded), not for section
   resolution (exact string equality after normalization, at either the
   full-heading or number-omitted-title level), and not for hierarchy scope
   (computed purely from parsed dotted-number tuple prefixes). No
   embeddings, no semantic similarity, and no lookup outside what was
   actually retrieved for this specific response are introduced by this
   policy.

---

## 8. Correct Abstention (unanswerable questions)

**Inputs used:** `question`, `generated_response`, `unanswerable_reason`,
`retrieved_context`.

- **PASS** — the response appropriately declines, or states that the
  requested information cannot be established from the available
  source/scope. **PASS does not require any specific wording or hedging
  style** — there is no required literal phrase such as "I cannot answer."
  A response such as "the available source does not specify the requested
  information" may PASS if it substantively avoids supplying an unsupported
  answer. Judge the behavior (does it avoid asserting an answer the source
  cannot support?), never a required phrase. A refusal may include a brief,
  accurate, supported explanation and still PASS — brevity is not required;
  fabrication is disqualifying (see Unsupported Answer, immediately below).
- **FAIL** — the response substantively answers a question that should not
  be answerable from the supplied source/scope.
- **REVIEW_REQUIRED** — whether the response constitutes a genuine answer
  versus an appropriate abstention (possibly phrased indirectly, e.g. "the
  available material does not specify X" rather than a direct "I can't
  answer") is genuinely ambiguous.

**Boundary lesson from human review (stated abstractly):** a response that
states a requested fact cannot be established, while *also* volunteering an
adjacent, true, supported piece of context, is still a valid abstention —
it should not be marked FAIL merely for including extra correctly-grounded
context alongside the decline.

## 9. Unsupported Answer (unanswerable questions)

- **PASS** — the response does not introduce unsupported substantive
  factual content.
- **FAIL** — the response fabricates or asserts unsupported material
  information (a confident-sounding answer to a question the source cannot
  support).
- **REVIEW_REQUIRED** — a material explanatory claim exists whose support
  cannot be determined from the supplied retrieved context.

**Do not conflate** "I cannot establish this from the available source"
(a correct abstention move) with an unsupported factual answer (a
fabrication) — these sit at opposite ends of this axis and must never be
scored the same way.

---

## 10. Final GAS — downstream deterministic composition (never LLM-guessed)

Final GAS is **never** requested from or produced by the LLM judge. It is
computed afterward, deterministically, from the finalized component labels
(the five semantic labels plus the separately-computed Citation Validity):

```
For an ANSWERABLE row, after inserting the deterministic Citation Validity
result alongside the five semantic labels:

  if ALL required component labels are PASS:
      GAS = PASS
  else if ANY required component label is FAIL:
      GAS = FAIL
  else if ANY required component label is REVIEW_REQUIRED
       (and none is FAIL):
      GAS = REVIEW_REQUIRED   # unresolved, pending human adjudication

For an UNANSWERABLE row:
      GAS = N/A               # safety is evaluated separately via
                               # Correct Abstention / Unsupported Answer
```

**Non-negotiable rules:**
- REVIEW_REQUIRED must never be silently converted to N/A, silently dropped
  from the GAS denominator, or silently treated as PASS or FAIL for
  reporting purposes. A REVIEW_REQUIRED GAS is a real, counted, reportable
  state meaning "not yet resolved."
- Once human adjudication resolves every REVIEW_REQUIRED component for a
  row, GAS can be recomputed to a final PASS/FAIL using the same
  composition rule above.
- Citation Validity is inserted into this composition from the
  deterministic evaluator (§7),
  never guessed by the LLM.

---

## 11. Potential human-label / rubric conflicts requiring adjudication

Found during rubric design review by comparing this rubric's rules against
`eval/annotation/final_human_labels_v2.1.csv` and the finalized human-review
notes. **Nothing was changed as a result of this review** — these are
flagged for human adjudication, not resolved unilaterally, and the rubric
was not weakened to force agreement with any of them.

### Conflict 1 — `q027`, Completeness — RESOLVED via formal gold revision (v1.2)

**Status: resolved. `q027` is no longer a live, unresolved rubric/gold
conflict.** This section is retained (rather than deleted) as a record of a
real gold-design inconsistency discovered during Judge rubric design
review, and of how it was corrected — not as an outstanding item requiring
further adjudication.

**What the conflict was.** The frozen gold for this question, at the time
of rubric design (`eval/questions_pilot_v1.1.jsonl`), listed **two**
required claims: (c01) a fact about a specific course's core-course status
for one program track, and (c02) a fact about which population a
substitution policy applies to. The finalized human label for both Control
and this question's Treatment response was Completeness **PASS**, with a
recorded rationale that only c02 is necessary for a minimally complete
answer to the specific question asked — c01 was judged unnecessary in
context, even though it remained listed in gold as a required claim. Per §4
of this rubric, the judge itself is forbidden from ever reaching that same
conclusion autonomously — applied literally to v1.1 gold, the judge would
have flagged this response `REVIEW_REQUIRED` (claim c01 missing, but
plausibly gold-design-inconsistent) rather than silently reproduce the
human's PASS. That divergence between frozen gold and the finalized human
interpretation was the conflict.

**How it was resolved.** This was corrected through **formal gold
versioning**, mirroring the precedent already established for `q007` in
v1.1: `eval/questions_pilot_v1.2.jsonl` removes the now-unnecessary required
claim (`q027-c01`) from `q027`'s gold_claims, retaining `q027-c02` verbatim.
This is a **resolved gold-design inconsistency**, corrected the same way as
the `q007` precedent, **not** a permanent judge exception and **not** a
hard-coded `q027` exception in the judge prompt — the judge prompt contains
no reference to this question at all. Full detail, including the
before/after claim sets and why generation was not rerun, is in
`eval/annotation/pilot_30q_v2.1/gold_revision_log.md`. This gold correction
caused no change to any metric value, since the finalized human labels
already reflected the interpretation gold v1.2 now formalizes.

**Why this matters for calibration going forward:** against gold v1.2, a
judge applying §4 literally to this question's response now reaches
Completeness PASS on its own — the same conclusion the human reviewer
reached — without needing to invoke `rubric_or_gold_conflict` or
REVIEW_REQUIRED for this question. No judge behavior change was needed; the
frozen requirement it must respect changed instead.

### Non-conflicts worth recording (rubric validated, not contradicted)

- **Scenario-specific numeric correctness.** One development question asks
  the responder to apply a general numeric rule to a specific stated
  scenario; gold resolves this to a single specific number, not a range.
  The finalized human label marks a response that instead restates the
  general range as Correctness FAIL. Initially this looked like it might
  expose an ambiguity in this rubric's Correctness/Completeness boundary,
  but checking gold directly shows gold already resolved the
  scenario-specific value — the response's range statement is materially
  inconsistent with that resolved value, not merely less complete. No
  rubric change needed; recorded here because the *general pattern*
  (scenario-application correctness vs. generic-rule restatement) is worth
  watching during calibration on new families.
- **Multi-part comparison question.** One development question asks for a
  comparison across two options/tracks; the response under review answers
  only one side. The finalized human label is Task Completion PASS,
  Completeness FAIL — exactly the separation this rubric's §2/§4 require.
  This is a clean positive validation of that design boundary, not a
  conflict.
- **Joint/implicit conveyance of a single required fact.** One development
  question's frozen gold requires a single specific fact; one response
  states two adjacent true, cited facts that only *jointly* imply the
  required fact rather than stating it directly, and the finalized human
  label treats this as sufficient (Completeness PASS). This is consistent
  with §4's "semantically equivalent wording is acceptable" clause, but is
  flagged in the calibration failure taxonomy (see calibration plan) as a
  likely source of judge/human disagreement, since detecting joint/implicit
  conveyance is harder than detecting direct restatement.

---

## 12. Ambiguous design decisions flagged for human review

1. ~~**Citation Support when Citation Validity fails.**~~ — **Resolved.**
   §6's interaction rule stands as a human-confirmed rubric decision, not an
   open ambiguity: when a citation is nonexistent or was not part of the
   retrieved context actually supplied to this response (Case A, §6/§7),
   Citation Validity = FAIL and Citation Support = FAIL for the material
   claim relying on that citation — not N/A. The alternative previously
   considered here (scoring Citation Support as N/A in that situation) was
   rejected: an invalid citation means the claim is *effectively* uncited
   for support purposes, which §6's own FAIL clause ("a material claim
   lacks any citation") already covers. No longer an open item.
2. **REVIEW_REQUIRED granularity for Task Completion — calibration item,
   not a blocker for the first development run.** Task Completion's enum
   includes REVIEW_REQUIRED (per the brief) but the axis has no natural N/A
   state the way Correctness/Faithfulness/Citation Support do. The current
   narrow definition (§2: REVIEW_REQUIRED reserved for genuinely ambiguous
   real-attempt-vs-evasion cases, not a default for short or imperfect
   responses) is sufficient to proceed with the first development run.
   Actual ambiguous cases encountered during calibration should be
   inspected and used to sharpen this boundary with worked examples, rather
   than blocking the run on a boundary that has not yet been exercised
   against real cases.
3. ~~**`q027`-style adjudication path**~~ — **Resolved.** §11 Conflict 1 was
   adjudicated via formal gold revision (`eval/questions_pilot_v1.2.jsonl`);
   see that section for detail. No longer an open item.
4. ~~**Whether Correct Abstention PASS requires the response to explicitly
   signal uncertainty.**~~ — **Resolved.** No literal refusal phrase (e.g.
   "I cannot answer") and no explicit hedging style is required for PASS. A
   response may PASS Correct Abstention if it clearly communicates that the
   requested information is not established by the available source and
   does not go on to provide an unsupported answer (§8: "the available
   material does not specify X"-style phrasing suffices on its own). This
   is now a human-confirmed rubric decision, not an open ambiguity — but it
   remains **methodologically resolved while empirically under-tested**:
   the development set contains zero Correct Abstention FAIL examples, so
   this boundary has not been stress-tested against a case where it
   actually breaks. That gap is a calibration/validation-coverage question
   (see `judge_calibration_plan_v1.md` §1/§5), not a reason to reopen the
   wording decision itself.

---

## 13. What this rubric explicitly does not do

- It does not compute Final GAS via the LLM.
- It does not ask the LLM about Citation Validity.
- It does not allow the LLM to see human labels, Human Notes, Failure Type,
  arm identity, verifier decisions, treatment strategy version, prior judge
  output, or aggregate metrics (see `judge_prompt_v1.txt` input contract).
- It does not treat this pilot's agreement statistics as evidence of
  production-grade reliability.
