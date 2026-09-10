# Gold Revision Log — Pilot 30-Question Dataset

## q007 — gold correction (v1.0 → v1.1)

- **Scope**: q007 only. No other question in `eval/questions_pilot_v1.1.jsonl`
  differs from `eval/questions_pilot_v1.0.jsonl` (verified byte-identical for
  all 29 other rows).
- **Question text and all experiment metadata unchanged**: `question`,
  `answerable`, `question_type`, `question_family_id`, `policy_unit_id`,
  `experiment_split`, `unanswerable_reason` are identical to v1.0.
- **What changed**: `gold_answer` and `gold_claims` (and their
  `supporting_evidence`) were revised.
- **Why**: the previous gold over-specified the answer by requiring the
  aggregate "no more than 12 units of undergraduate coursework" cap
  (`q007-c01`) as a claim necessary for a minimally complete answer. Human
  review determined this aggregate-cap fact was not necessary to answer the
  specific question asked ("What conditions and limits apply if I want a
  6-unit upper-division undergraduate course to count..."), which is about
  conditions/limits for a specific course, not the aggregate program-wide cap.
- **Revised required gold content** (`q007-c02` through `q007-c05`, retained
  from v1.0 verbatim — only `q007-c01` was removed):
  1. Undergraduate units count against the 36-unit cap on courses outside
     Heinz College.
  2. The student must submit a general petition and receive approval before
     the class starts for the undergraduate elective to count.
  3. Generally only upper-division undergraduate classes are considered.
  4. A grade of B or better is required for the approved undergraduate units
     to count toward the degree.
- **The 12-unit undergraduate cap remains true handbook context** (section 6
  of the handbook genuinely states it) — it was simply removed as a
  *required* claim for this specific question's completeness grading.
- **Generation was NOT rerun.** v1, v2, and v2.1 pilot generation (Control and
  Treatment responses) all used `eval/questions_pilot_v1.0.jsonl` — the
  original, uncorrected gold. No response text or retrieval output was
  changed as a result of this gold revision.
- **Final human evaluation uses gold version v1.1** (this corrected version)
  — i.e., q007's Completeness/GAS labels in the final merged workbook were
  judged against the v1.1 gold_claims (4 required claims), not the original
  5.
- **Validation**: `eval/questions_pilot_v1.1.jsonl` passes
  `heinzy.eval.dataset.validate_dataset()` with 0 errors (30/30 rows valid).

## q027 — history: human LABEL correction (v1), then gold correction (v1.1 → v1.2)

This entry preserves the full historical sequence for q027, in the order it
actually happened. Do not read only the final step below in isolation — the
gold correction was a *formalization* of an interpretation the human review
had already reached, not a new interpretation.

### Step 1 — human LABEL correction (no gold change at the time)

- At the time this correction was made, **no gold data was changed for
  q027.**
- Completeness and Final GAS were corrected to **PASS** for both Control
  (response_id R030) and v1 Treatment (response_id R040), because human
  review determined the response sufficiently resolves the user's
  BIDA/non-BIDA Capstone-substitution question, even though the then-current
  gold (`eval/questions_pilot_v1.1.jsonl`) listed a second required claim
  (`q027-c01`, a fact about a specific program track's core-course status)
  that the response did not convey. The original draft labels had marked
  these FAIL; the correction is recorded via the "Human-reviewed correction"
  notes already present on R030/R040 in
  `eval/annotation/pilot_30q/annotation_workbook_human_reviewed_v1.xlsx`.
- At the time, this was recorded purely as a human labeling-judgment
  correction, distinguished from a gold ground-truth correction (q007's) —
  i.e. gold and the finalized human label were, at that point, inconsistent
  with each other. This inconsistency was subsequently identified as
  Judge-rubric-design "Conflict 1" (`eval/judge/judge_rubric_v1.md` §11)
  during LLM-as-a-Judge design review.

### Step 2 — gold correction (v1.1 → v1.2), formalizing Step 1's interpretation

- **Scope**: q027 only. No other question in
  `eval/questions_pilot_v1.2.jsonl` differs from
  `eval/questions_pilot_v1.1.jsonl` (verified byte-identical for all 29
  other rows).
- **Question text and all experiment metadata unchanged**: `question`,
  `answerable`, `question_type`, `question_family_id`, `policy_unit_id`,
  `experiment_split`, `unanswerable_reason` are identical to v1.1.
- **What changed**: `gold_answer` and `gold_claims` were revised. Required
  claim `q027-c01` ("MISM-BIDA requires 94-739 Capstone Project as a core
  course") was **removed** from the required gold claim set. Required claim
  `q027-c02` ("The Lean Innovation Lab substitution is offered to MISM
  (non-BIDA) students") was **retained verbatim**, with its original
  `claim_id`. `gold_answer` was reworded to be consistent with the
  narrowed required-claim set, while preserving the same substantive
  conclusion (a BIDA student may not use this substitution).
- **Why**: this is a **GOLD CONSISTENCY correction, not a model-performance
  improvement.** The previous gold over-specified the minimally complete
  answer to this specific question by requiring a claim about MISM-BIDA's
  core-course list (`q027-c01`) that is not necessary to answer the question
  actually asked ("Can I [a BIDA student] take Lean Innovation Lab instead of
  the 94-739 Capstone Project?") — that question is fully and correctly
  resolved by stating which population the substitution policy applies to
  (`q027-c02`) and that a BIDA student falls outside it. This mirrors the
  precedent already established for q007 in v1.1 (above): an aggregate or
  contextual fact that is true but not required for a minimally complete
  answer to the specific question asked was removed from the required claim
  set, not because it was false, but because it over-specified completeness.
- **Generation was NOT rerun.** Generation (Control, v1 Treatment, v2.1
  Treatment) used `eval/questions_pilot_v1.0.jsonl` throughout — the
  original, pre-q007-correction, pre-q027-correction gold. No response text
  or retrieval output was changed as a result of this gold revision.
- **Final human-evaluation interpretation now uses gold version v1.2**: the
  q027 Completeness/GAS PASS labels recorded in Step 1 above are, as of this
  correction, consistent with the frozen gold requirements (only `q027-c02`
  is required), rather than being a label-level override of a still-broader
  gold.
- **No aggregate metric change results from this correction.** The
  finalized human labels for q027 (Step 1) already reflected the
  interpretation gold v1.2 now formalizes — PASS for Completeness/Final GAS,
  for both Control and Treatment. Gold v1.2 does not change what any
  response was judged to convey; it changes only which claims are recorded
  as required, to match a judgment already made. See
  `human_eval_metrics_v2.1.json` (`evaluation_gold_version: "v1.2"`) — every
  `metrics`/`paired_gas_transitions` value is unchanged from before this
  correction.
- **Validation**: `eval/questions_pilot_v1.2.jsonl` passes
  `heinzy.eval.dataset.validate_dataset()` with 0 errors (30/30 rows valid).
- **Resolution recorded in the Judge design docs**: `eval/judge/judge_rubric_v1.md`
  §11 no longer lists q027 as a live, unresolved conflict; it is documented
  there as a resolved gold-design inconsistency, corrected through this
  formal gold versioning step (option (a) of the two adjudication paths that
  section had previously left open), not through a hard-coded judge
  exception.

## q007 consistency check (post gold-correction)

The v1 Control (R036) and v1 Treatment (R041) responses to q007 both still
omit the B-or-better requirement (`q007-c05`, retained in v1.1's gold). Their
existing Completeness=FAIL and Final GAS=FAIL labels therefore remain valid
under the v1.1 gold and were **not** changed as a result of the gold
correction — only the *reason* a hypothetical future response might fail
Completeness narrowed from 5 possible missing claims to 4.
