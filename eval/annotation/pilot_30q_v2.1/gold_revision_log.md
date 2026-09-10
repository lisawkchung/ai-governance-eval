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

## q027 — human LABEL correction (not a gold-data correction)

- **No gold data was changed for q027.**
- Completeness and Final GAS were corrected to **PASS** for both Control
  (response_id R030) and v1 Treatment (response_id R040), because human
  review determined the response sufficiently resolves the user's
  BIDA/non-BIDA Capstone-substitution question. The original draft labels
  had marked these FAIL; the correction is recorded via the "Human-reviewed
  correction" notes already present on R030/R040 in
  `eval/annotation/pilot_30q/annotation_workbook_human_reviewed_v1.xlsx`.
- This log entry exists purely for provenance/audit clarity, distinguishing
  a human labeling-judgment correction (q027) from a gold ground-truth
  correction (q007).

## q007 consistency check (post gold-correction)

The v1 Control (R036) and v1 Treatment (R041) responses to q007 both still
omit the B-or-better requirement (`q007-c05`, retained in v1.1's gold). Their
existing Completeness=FAIL and Final GAS=FAIL labels therefore remain valid
under the v1.1 gold and were **not** changed as a result of the gold
correction — only the *reason* a hypothetical future response might fail
Completeness narrowed from 5 possible missing claims to 4.
