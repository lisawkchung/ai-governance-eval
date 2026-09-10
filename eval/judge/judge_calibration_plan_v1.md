# Heinzy LLM-as-a-Judge — Calibration Plan v1

**Status: DESIGN ONLY. No judge has been run. No calibration pass has been
performed. No dev/holdout split has been created or frozen.**

This plan defines *how* Judge Prompt v1 will be calibrated and, later,
independently validated. It does not execute any of these steps.

---

## 1. What "the pilot" is for, and is not for

The existing 30-question / 60-response pilot
(`eval/questions_pilot_v1.1.jsonl`, `eval/annotation/final_human_labels_v2.1.csv`)
is **judge-development data only**.

- **Is for:** debugging the rubric and prompt, finding cases where the
  judge's literal rules produce a surprising or wrong-feeling label,
  iterating on wording, and building the failure-mode taxonomy in §3 below.
- **Is not for:** claiming any measured agreement rate as evidence of
  production-grade judge reliability, and is not to be split into a "dev"
  and "holdout" partition of itself — every family in it is, by
  construction, either directly rubric-exposed or too thin in independent
  FAIL coverage to serve as a holdout (full detail:
  `eval/annotation/judge_split_feasibility_audit_v1.md`).

Concrete limitation reminder, carried from the feasibility audit:

| Axis | Independent FAIL families available in the pilot |
|---|---|
| Faithfulness | 0 |
| Correct Abstention | 0 |
| Unsupported Answer | 0 |
| Correctness | 0 |
| Task Completion | 0 (once known rubric-development exposure is honored) |
| Completeness | 1 |
| Citation Validity / Citation Support | 1 (same single case for both) |

**High aggregate agreement on this set does not establish failure-detection
reliability**, because the set has almost no independent failure examples
to fail to detect. This sentence should be repeated in any report that
cites a pilot-stage agreement number.

---

## 2. Development procedure (uses the 30-question pilot only)

1. Run Judge Prompt v1 (Variant A/B as appropriate) against all 60
   development responses.
2. Compare each judge label against the corresponding finalized human label
   in `eval/annotation/final_human_labels_v2.1.csv`, axis by axis.
3. For every disagreement, and for every judge `REVIEW_REQUIRED`, inspect
   manually: read the judge's `reason`/claim-list fields against the actual
   response, retrieved context, and gold.
4. Classify each disagreement using the failure taxonomy in §3.
5. Revise the prompt/rubric wording (never the human label, never gold)
   to address genuine judge misunderstandings; re-run only on development
   data.
6. Repeat until disagreements are either resolved by a rubric-wording fix
   or explicitly logged as an open rubric ambiguity for human review
   (`judge_rubric_v1.md` §12) rather than being “fixed” by narrowing the
   rubric to match one specific case.

**Anti-pattern to avoid explicitly:** editing the rubric so that it happens
to reproduce a specific pilot label, rather than because the underlying
principle was actually wrong. If a rubric change would only affect one
known pilot row and no general principle, that is a signal to log it as an
ambiguity, not to "fix" it silently.

---

## 3. Judge failure-mode taxonomy

Use this fixed vocabulary when classifying a disagreement, so failure
patterns are comparable across calibration rounds:

- `overly_generous_on_incomplete` — judge PASSes Completeness when a
  required claim is genuinely missing.
- `overly_strict_on_concise` — judge FAILs Completeness/Task Completion for
  a short answer that is actually sufficient.
- `confuses_correctness_with_completeness` — judge conflates "said
  something wrong" with "left something out," or vice versa.
- `uses_gold_to_judge_faithfulness` — judge's Faithfulness reasoning
  references gold_answer/gold_claims/supporting_evidence instead of
  retrieved_context only.
- `accepts_unsupported_generalization` — judge misses an unsupported
  exclusivity/prohibition/universality inference that a positive statement
  does not itself establish.
- `misses_unsupported_citation` — judge PASSes Citation Support for a
  citation that does not actually back its attached claim.
- `mishandles_refusals` — judge misapplies Task Completion / Correct
  Abstention logic to a refusal or partial-refusal response.
- `mishandles_na` — judge assigns N/A where a real material claim exists,
  or fails to assign N/A where no material claim exists.
- `overuses_review_required` — judge flags REVIEW_REQUIRED for cases that
  are actually resolvable from the supplied material.
- `underuses_review_required` — judge guesses PASS/FAIL where the supplied
  material is genuinely insufficient.
- `verbosity_bias` — judge's label or reasoning is influenced by response
  length/style rather than substance.
- `mirrors_gold_wording_too_literally` — judge FAILs Completeness for a
  semantically equivalent but differently-worded response, or fails to
  recognize a joint/implicit statement of a required claim.
- `redesigns_gold_requirements` — judge silently treats a gold_claims entry
  as unnecessary without setting `rubric_or_gold_conflict` and returning
  REVIEW_REQUIRED. **This is a hard-stop bug, not a calibration nuance** —
  any occurrence must block further calibration until fixed.
- `arm_leakage_suspected` — judge's reasoning references anything that
  looks like inferred Control/Treatment identity (should be structurally
  impossible given the input contract, but check for it anyway).

---

## 4. Metrics to compute during and after calibration

Computed against development data during calibration (informative only,
never reported as reliability evidence), and again against the future
validation sets in §5 (where they become meaningful reliability estimates):

- Raw agreement per axis (judge label == human label).
- Confusion matrix per axis (all 3–4 enum values, not just PASS/FAIL).
- FAIL precision / recall / F1 **only where the underlying data has enough
  negative (FAIL) examples for the number to mean anything** — on the
  current pilot this is explicitly not the case for most axes (§1 table);
  do not compute or report a FAIL recall number from 0 or 1 examples and
  present it as if it were a statistic.
- Cohen's kappa **only where the class distribution permits meaningful
  interpretation** — kappa is unstable and easily misleading on
  heavily skewed, small-N distributions like the current pilot's.
- REVIEW_REQUIRED count and rate per axis, valid-N/A count and rate where
  applicable, automatic judgment coverage, and execution/parsing/schema
  error count and rate — all defined precisely, and required in every
  report, in §4.1 below.
- Human-adjudication rate (§4.1/§4.2.C).
- Four distinct agreement quantities (judge-only, coverage, adjudication
  rate, post-adjudication pipeline result) — never collapsed into one
  number — defined in §4.2.

**Standing statement, to be repeated wherever any of the above numbers are
reported from development data:** *high PASS-heavy agreement does not
establish failure-detection reliability.*

### 4.1 Per-axis outcome accounting (every development and validation
    report, no exceptions)

For each axis, every applicable row must land in exactly one of five
mutually exclusive outcome buckets — never blended together, never
collapsed into fewer categories for convenience:

- **PASS**
- **FAIL**
- **N/A** — only a valid bucket for axes whose schema permits it
  (Correctness, Faithfulness, Citation Support); not a valid bucket for
  Task Completion, Completeness, Correct Abstention, or Unsupported Answer.
- **REVIEW_REQUIRED**
- **Execution/parsing/schema error** — the judge call failed to execute,
  returned invalid JSON, failed schema validation against
  `judge_output_schema_v1.json`, or otherwise produced no valid labeled
  output at all for this row/axis.

**These five buckets are not interchangeable, and two conflations are
explicitly forbidden:**
- An execution/parsing/schema error is **not** a REVIEW_REQUIRED and must
  never be silently converted into one, silently dropped from any
  denominator, or excluded from reporting. A parsing/schema failure means
  the judge produced **no usable judgment at all**; REVIEW_REQUIRED means
  the judge ran successfully and explicitly declined to resolve the case.
  Conflating the two hides a reliability problem (crashes or malformed
  output) inside what would otherwise look like a principled calibration
  signal.
- **Valid N/A is a legitimate automatic judge resolution**, not an
  abstention, for axes whose schema permits it — correctly determining "no
  material claim exists" is itself a resolved judgment.

**Definitions:**

- `applicable_N` — the number of rows this axis applies to at all (e.g.
  Correctness/Faithfulness/Citation Support apply only to answerable rows;
  Correct Abstention/Unsupported Answer only to unanswerable rows).
- `automatic_resolved_N = PASS + FAIL + valid N/A` (N/A included only for
  axes where it is a valid schema state).
- `automatic_judgment_coverage = automatic_resolved_N / applicable_N`.
- REVIEW_REQUIRED cases are, by definition, **not** automatically resolved
  and must never be added into `automatic_resolved_N`.
- Execution/parsing/schema errors are likewise **not** automatically
  resolved (there is no output to resolve) and must never be added into
  `automatic_resolved_N` — they get their own separate count/rate, never
  folded into REVIEW_REQUIRED and never folded into `automatic_resolved_N`.

**Do NOT define automatic judgment coverage as `1 - REVIEW_REQUIRED
rate`.** That formula silently (a) treats every execution/parsing/schema
error as if it were an automatic resolution, which is wrong, and (b)
ignores that valid N/A is itself a legitimate automatic resolution for
axes that support it, undercounting coverage on those axes.

**Mandatory reporting, per axis (never only in aggregate):**

- REVIEW_REQUIRED count and rate (`REVIEW_REQUIRED / applicable_N`).
- Valid N/A count and rate, where N/A is valid for that axis.
- Automatic judgment coverage, as defined above.
- Execution/parsing/schema error count and rate (`error_N / applicable_N`),
  reported as its own category — never merged into REVIEW_REQUIRED, never
  excluded from the report.
- Human adjudication rate (fraction of `applicable_N` that required a
  human to resolve a REVIEW_REQUIRED judge output to a final label).

### 4.2 Agreement metrics — four distinct, separately reported quantities

Do not describe any single number as "judge accuracy" or "judge-human
agreement" without specifying which of the following four it is. These
measure different things and must never be collapsed into one number or
substituted for one another:

**A. Judge-only agreement** — agreement between the human label and the
judge's own output, computed **only** over rows the judge itself resolved
to PASS/FAIL/valid-N/A without any human intervention (i.e. over
`automatic_resolved_N`, §4.1). This is the only one of the four that
isolates the judge's own standalone reliability.

**B. Automatic judgment coverage** — the fraction of `applicable_N` the
judge resolved on its own (§4.1's `automatic_judgment_coverage`). Not an
agreement number at all — a denominator-size statement, always reported
alongside (A), never instead of it.

**C. Human adjudication rate** — the fraction of `applicable_N` that
required human review to reach a final label (§4.1).

**D. Post-adjudication evaluation-pipeline result** — the final label
distribution (and, where relevant, Final GAS agreement) computed **after**
a human has resolved every REVIEW_REQUIRED case in the reported set to a
final PASS/FAIL, per `judge_rubric_v1.md` §10.

**Explicit statement, required wherever (D) is reported:** *(D) is a
HUMAN-IN-THE-LOOP PIPELINE result and must not be presented as evidence of
standalone Judge accuracy.* It measures what the full pipeline (judge +
human adjudication) produces together, not what the judge can do alone —
only (A) measures that.

**Preserve the original raw judge output.** Human adjudication produces a
final label for reporting purposes (D); it must never overwrite, discard,
or be conflated with the judge's own original REVIEW_REQUIRED output, which
remains the record of what the judge actually produced on its own (needed
to correctly compute A/B/C and to audit calibration decisions later).

**Execution/parsing/schema failures must never be silently converted into
REVIEW_REQUIRED, silently converted into a guessed PASS/FAIL, or excluded
from any of A-D's denominators or from the mandatory reporting in §4.1.**
They are a distinct failure category from REVIEW_REQUIRED and must remain
visible, with their own count and rate, in every report.

**Mandatory warning, to be included verbatim or in substance wherever any
of A/B/C/D is reported:** *High agreement on the auto-resolved subset (A)
must NOT be presented alone, and must never substitute for (D) or vice
versa, if a material share of difficult cases was sent to
REVIEW_REQUIRED.* A judge that resolves only the easy 70% of cases on its
own and agrees with humans 98% of the time on that 70% (A) has not thereby
shown 98% reliability overall — the remaining 30% is exactly where
disagreement and genuine difficulty concentrate. Likewise, a high
post-adjudication pipeline result (D) reflects the pipeline including
human correction, not the judge working alone, and must always be labeled
as such when reported.

---

## 5. Future independent validation plan (design only — no examples created)

The judge prompt will be **frozen before** any validation example is
generated or reviewed for judge tuning purposes, to prevent validation
leakage into development. Two distinct, separately-reported components:

### A. Natural response validation

**Purpose:** estimate judge/human agreement on responses the system
produces *naturally*, on new material, without deliberately engineering
failure cases.

**Requirements:**
- New question families only — never a new phrasing of an
  already-development-exposed family (`judge_rubric_v1.md` uses the term
  "development-exposed"; see the feasibility audit for the exact exposed
  set: `q001, q004, q007, q010, q019, q021, q025, q027, q030` and their
  families).
- Generate responses through the normal system pipeline (Control and
  Treatment), unmodified.
- Human-label these responses independently, using the same frozen rubric
  the judge uses, before running the judge (to avoid the human anchoring on
  the judge's output).
- Run the frozen judge; compare against the independent human labels.

**Interpretation:** this estimates judge reliability under the distribution
of responses the system actually produces in practice — mostly PASS-heavy,
same as the pilot, but on families the judge/rubric has never seen.

### B. Targeted challenge validation

**Purpose:** specifically test whether the frozen judge can *detect*
important failure modes, since natural-response validation alone will
likely still be PASS-heavy and under-power failure-detection estimates
(exactly the problem with the current pilot).

**Requirements:**
- New question families (never reused from development or from natural
  validation).
- Deliberately selected or constructed response cases covering, at minimum:
  incorrect factual claim; missing required claim; unsupported factual
  claim; unsupported generalization/exclusivity inference; citation present
  but not supporting its claim; uncited material claim; semantic task
  non-completion; false refusal; unanswerable question answered anyway;
  fabricated content on an unanswerable question.
- Each of these should ideally come from a **different** underlying
  question family than the others, so no single family's idiosyncrasies
  are mistaken for a general failure-detection capability.

**Reporting constraint (do not violate this):** challenge-set aggregate
agreement must be reported **separately** from natural-response validation
agreement, and must never be presented as "production-like judge accuracy."
A judge that does well on a challenge set built specifically to contain
each failure mode once is not thereby shown to perform well on the natural,
mostly-PASS distribution the system produces day to day, and vice versa.

### Sizing — deliberately not fixed here

An initial batch of **8–12 new independent question families** (matching
the gap estimate in the feasibility audit) is a *possible pilot-sized
starting point* for validation construction, not a claimed-sufficient
number. Before building it:

1. **Predefine target failure mechanisms** — the list in §5.B above, plus
   any additional mechanism identified during development calibration
   (§2–§3).
2. **Predefine minimum desired negative (FAIL) coverage per axis** — e.g.
   "at least N independent families producing a genuine Faithfulness FAIL"
   — a specific number to fill in during actual validation design, not
   guessed here.
3. **Predefine family-independence rules** — no shared `question_family_id`
   with any development-exposed family or with each other; no reuse of the
   same underlying policy/retrieval scenario merely reworded.
4. **Predefine judge acceptance criteria** — what agreement/FAIL-recall
   profile would be considered "good enough to trust in production" — left
   open here deliberately; this needs human and/or statistical design input
   once real validation data exists, not a number invented in advance of
   any evidence.
5. Only **after** 1–4 are agreed, construct the validation set, run
   coverage analysis by axis/failure mechanism against the predefined
   targets, and **expand** validation data for any axis still
   under-covered before treating the judge as validated on that axis.

No numeric acceptance threshold is set in this document. Setting one now,
without evidence, would be exactly the kind of unsupported claim this whole
plan exists to prevent.

---

## 6. Pilot / validation / confirmatory separation (do not blur these)

| Stage | Data | Purpose |
|---|---|---|
| **Current pilot** | 30 questions / 60 responses, `questions_pilot_v1.1.jsonl` | System development, Treatment-strategy development (v1→v2→v2.1), **judge development** |
| **Future judge validation** (this plan, §5) | New question families, never yet created | Evaluates the *evaluator's* reliability (judge vs. human agreement) |
| **Future confirmatory experiment** | The untouched confirmatory dataset, not yet used for anything in this project | Evaluates the actual Control-vs-Treatment product hypothesis, using the by-then-validated judge |

**Hard rule:** confirmatory-dataset examples must never be used for judge
prompt tuning, at any stage, for any reason — including "just to check" the
judge on a handful of confirmatory rows during development. Once a
confirmatory example has been looked at for judge-tuning purposes, it can
no longer serve its confirmatory role. This project does not use the
confirmatory dataset for anything until explicitly and separately approved,
and this calibration plan does not request that approval.

---

## 7. Explicit non-goals of this plan

- Does not run a judge.
- Does not create validation examples.
- Does not create or freeze a judge_dev/holdout split of the current pilot
  (none is possible with adequate independent FAIL coverage — see §1).
- Does not set a numeric production acceptance threshold.
- Does not touch the confirmatory dataset.
