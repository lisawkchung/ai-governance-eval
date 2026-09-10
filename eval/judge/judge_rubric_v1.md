# Heinzy RAG Evaluation — LLM-as-a-Judge Rubric v1

**Status: DESIGN / DEVELOPMENT CANDIDATE. No judge has been run against
this rubric. No dev/holdout split exists yet.** The explicit design
contradictions identified during initial rubric review (Correctness PASS
boundary, Citation Validity/Citation Support interaction, Task Completion
REVIEW_REQUIRED overuse risk, Correct Abstention phrasing rigidity, and the
`q027` gold/label conflict — §11) have been resolved ahead of the first
development run. This does **not** mean the judge is validated, calibrated,
or production-ready — see §0 below and `judge_calibration_plan_v1.md`.

This document is the human-readable rubric backing `judge_prompt_v1.txt` and
`judge_output_schema_v1.json`. It exists so a reviewer can inspect the
grading logic without reading the raw prompt, and so future prompt revisions
have a stable reference to diff against.

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
`gold_claims`, `supporting_evidence`. **Never** outside knowledge.

- **PASS** — every material factual claim in the response is *verifiable*
  from the supplied correctness references, **and** every such claim is
  correct. Both conditions are required: PASS is not available merely
  because no claim was *contradicted* — an unverifiable claim blocks PASS
  just as a contradicted one does (see REVIEW_REQUIRED below; do not resolve
  an unverifiable claim to PASS or to FAIL).
- **FAIL** — at least one material factual claim is contradicted by, is
  inconsistent with, or materially misapplies the supplied correctness
  references (including applying a correct general rule incorrectly to a
  specific stated scenario).
- **N/A** — no material factual claim is made at all (e.g., a pure refusal).
  Do not use N/A for a claim that exists but cannot be verified — that is
  REVIEW_REQUIRED, not N/A; these two states must remain distinct.
- **REVIEW_REQUIRED** — a material claim is made, but the supplied gold and
  supporting evidence are insufficient to verify or contradict it. This is
  the **only** correct disposition for an unverifiable material claim: do
  not PASS it, and do not FAIL it merely because it is absent from gold.
  List every such claim in `unverifiable_claims`.

**The two-sided guardrail (memorize this, it is the most common failure
mode for this axis):** absence from gold does **not** automatically mean a
claim is false, and inability to contradict a claim does **not**
automatically mean it is correct. When a material claim cannot be judged
from the supplied references, the answer is REVIEW_REQUIRED — never a
guess in either direction.

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
- **Hierarchical section ambiguity** — a citation to a parent section (e.g.
  a top-level heading) when only a child subsection was retrieved, or vice
  versa, needs an explicit, documented resolution rule, not an implicit
  "close enough" heuristic.
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
