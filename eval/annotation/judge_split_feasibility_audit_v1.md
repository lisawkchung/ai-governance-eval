# LLM-as-a-Judge Calibration Split — Feasibility Audit v1

**Scope: feasibility audit only.** No judge was run, no judge prompt was written, no human label or gold datum was changed, no experiment was rerun, no significance testing was performed, and **no dev/holdout split was created or frozen**.

## Inputs read

- `eval/questions_pilot_v1.1.jsonl` (30 questions, 29 unique `question_family_id`s)
- `eval/annotation/final_human_labels_v2.1.csv` (60 rows: 30 Control + 30 Treatment)
- `eval/annotation/pilot_30q_v2.1/human_eval_metrics_v2.1.json`
- `eval/annotation/pilot_30q_v2.1/gold_revision_log.md`
- Local human-reviewed workbooks under `eval/annotation/` (inspected read-only, for Human Notes provenance)

## 1. Known rubric-exposed question_ids and families

**Given exposed (per instruction):** `q007, q010, q019, q027, q030`

**Family-forced additional exposure:** `q001` — it shares `question_family_id = mism_normal_courseload` with `q010`. This is the *only* family with more than one question in the whole pilot, so it's the only place family-forcing adds anything.

**→ Mandatory exposed set: `q001, q007, q010, q019, q027, q030`** (6 questions, 5 families).

**Additional candidates found in local provenance, reported separately with evidence (not silently merged):**

| qid | family | evidence |
|---|---|---|
| `q021` | `unit_increase_to_60_cqpa` | Verbatim-identical calibration text on response_ids R025/R060: *"this is a pure refusal on an answerable question after retrieval returned no chunks... Final GAS fails."* — a **general rule** applied across cases, not a one-off judgment. |
| `q025` | `internship_reporting_requirements` | Same verbatim rule, on R016/R035. |
| `q004` | `mism_bida_core_elective_units` | Notes on R051/R059 describe the "incomplete comparison" pattern; this question was also the repeated worked example for the "retrieval-limited failure" category used to design Verifier Prompt v2, and was independently re-inspected as a named development case across the v1→v2 and v2→v2.1 comparison tasks. |

I did **not** flag the remaining questions carrying Human Notes (e.g. `q008, q022, q023`) as exposed — their notes apply already-frozen definitions to a specific case without formulating or revising a general rule, per the instruction not to treat annotation alone as exposure.

**If the candidates above are honored** (recommended, given the strength of evidence): **9 questions / 8 families** are exposed in total: `q001, q004, q007, q010, q019, q021, q025, q027, q030`.

## 2. Per-axis PASS/FAIL/N/A counts (60 rows)

| Axis | PASS | FAIL | N/A | Unique FAIL question_ids | Unique FAIL families |
|---|---|---|---|---|---|
| Task Completion | 46 | 4 | 10 | q021, q025 | 2 |
| Correctness | 45 | 1 | 14 | q019 | 1 |
| Completeness | 39 | 11 | 10 | q002, q004, q007, q010, q019, q021, q025 | 7 |
| Faithfulness | 46 | **0** | 14 | — | 0 |
| Citation Validity | 44 | 2 | 14 | q013 | 1 |
| Citation Support | 44 | 2 | 14 | q013 | 1 |
| Final GAS | 37 | 13 | 10 | q002, q004, q007, q010, q013, q019, q021, q025 | 8 |
| Correct Abstention | 10 | **0** | 50 | — | 0 |
| Unsupported Answer | 10 | **0** | 50 | — | 0 |

## 3. Per-axis FAIL-family exposure and independent-validation verdict

| Axis | FAIL families | How many independent of exposure? | Can holdout validate this axis? |
|---|---|---|---|
| Task Completion | unit_increase_to_60_cqpa (q021), internship_reporting_requirements (q025) | **0** if candidates honored (both are candidate-exposed); 2 (thin, same failure mode) if not | **NO** (recommended reading) / marginal otherwise |
| Correctness | ai_management_demystifying_elective_units (q019) | **0** — mandatory-exposed | **NO** |
| Completeness | 7 families total | **1** — `mism_curriculum_requirements` (q002) | **PARTIALLY / MARGINAL** — one family only |
| Faithfulness | none | n/a | **NO** — zero FAIL examples exist anywhere, exposed or not |
| Citation Validity | bida_transfer_to_mism (q013) | **1** | **PARTIALLY / MARGINAL** — one family, and it can only sit on one side of a split |
| Citation Support | bida_transfer_to_mism (q013) — same case as above | **1** | **PARTIALLY / MARGINAL** — same family; Validity and Support have never failed independently of each other in this pilot |
| Final GAS | 8 families total | **2** — q002, q013 | **PARTIALLY / MARGINAL** — covers only 2 of ~5 distinct failure modes seen |
| Correct Abstention | none | n/a | **NO** — zero FAIL examples exist |
| Unsupported Answer | none | n/a | **NO** — zero FAIL examples exist |

**Explicit answers to the required question ("can both dev and independent validation contain meaningful FAIL examples without family leakage?"):**
- Faithfulness, Correct Abstention, Unsupported Answer: **impossible** — there is nothing to detect, in dev or holdout, because zero FAIL examples exist in the entire 60-row pilot for these axes.
- Correctness: **impossible for holdout** — its one FAIL example is rubric-exposed, so it can only ever be seen by judge_dev.
- Task Completion: **impossible for holdout** once the two candidate-exposed families are honored (both of its FAIL examples belong to them).
- Completeness, Citation Validity, Citation Support: **not simultaneously possible** — each has exactly 1 independent FAIL family. A family cannot be split across judge_dev and holdout without leakage, so whichever side does *not* receive that single family sees zero FAIL examples for that axis. You can put the example in dev (help calibrate the rubric, but then holdout can't confirm it) or in holdout (confirm once, but dev never saw a real example to calibrate against) — never both.

## 4. Family-level detail (families carrying any FAIL/N/A signal, or exposure)

| Family | question_ids | Answerable / Unanswerable | Exposed? | Control GAS | v2.1 Treatment GAS | Axes with FAIL/N/A |
|---|---|---|---|---|---|---|
| mism_normal_courseload | q001, q010 | 2 / 0 | mandatory | PASS, FAIL | PASS, PASS | completeness:FAIL |
| undergraduate_elective_credit | q007 | 1 / 0 | mandatory | FAIL | PASS | completeness:FAIL |
| ai_management_demystifying_elective_units | q019 | 1 / 0 | mandatory | FAIL | PASS | completeness:FAIL, correctness:FAIL |
| bida_capstone_substitution_lean_innovation | q027 | 1 / 0 | mandatory | PASS | PASS | (none — corrected to PASS) |
| concentration_declaration_gpa | q030 | 0 / 1 | mandatory | N/A (unanswerable) | N/A (unanswerable) | all answerable axes N/A (by design) |
| unit_increase_to_60_cqpa | q021 | 1 / 0 | **candidate** | FAIL | FAIL | task_completion:FAIL, completeness:FAIL, others N/A |
| internship_reporting_requirements | q025 | 1 / 0 | **candidate** | FAIL | FAIL | task_completion:FAIL, completeness:FAIL, others N/A |
| mism_bida_core_elective_units | q004 | 1 / 0 | **candidate** | FAIL | FAIL | completeness:FAIL |
| mism_curriculum_requirements | q002 | 1 / 0 | **none — independent** | FAIL | FAIL | completeness:FAIL |
| bida_transfer_to_mism | q013 | 1 / 0 | **none — independent** | FAIL | FAIL | citation_validity:FAIL, citation_support:FAIL |
| concentration_declaration_deadline | q009 | 0 / 1 | none | N/A | N/A | (unanswerable, correctly abstained both arms) |
| distributed_systems_current_instructor | q029 | 0 / 1 | none | N/A | N/A | (unanswerable, correctly abstained both arms) |
| mism_tuition | q028 | 0 / 1 | none | N/A | N/A | (unanswerable, correctly abstained both arms) |
| msppm_graduation_units | q008 | 0 / 1 | none | N/A | N/A | (unanswerable, correctly abstained both arms) |

The remaining 15 of 29 families contain only clean PASS rows on every axis for both arms and carry no FAIL/N/A signal at all.

**20-dev / 10-holdout headcount check:** After forcing the 8 mandatory+candidate exposed families (9 questions) into judge_dev, 21 clean families/questions remain, which can be split roughly 11/10 to reach an approximate 20/10 total. **This is arithmetically possible.** It is not, by itself, evidence that the holdout would contain anything meaningful to validate — see §3.

## 5. Feasibility conclusions

- **A. Can the pilot support judge DEVELOPMENT?** **FEASIBLE.** 60 labeled responses, real (if concentrated) FAIL signal across several axes, and 17 existing human-reviewer rationale notes already documenting rubric boundary-case reasoning (zero-hit false refusal, scenario-specific reasoning, retrieval-limited comparison, exclusivity wording). Enough to start drafting a judge prompt.
- **B. Can it support an INDEPENDENT judge HOLDOUT?** **PARTIALLY FEASIBLE**, headcount only. See C for why this doesn't translate into meaningful axis coverage.
- **C. Can that holdout meaningfully test failure detection per axis?** **NO for Faithfulness, Correctness, Correct Abstention, Unsupported Answer** (zero or fully-exposed FAIL examples). **NO for Task Completion** if the recommended candidate-exposure reading is used. **PARTIALLY / MARGINAL for Completeness, Citation Validity, Citation Support** (1 independent family each — thin, single-point).
- **D. Can it validate final judge-derived GAS?** **PARTIALLY FEASIBLE / THIN.** Only 2 of 8 GAS-FAIL-carrying families (q002, q013) are independent, covering only 2 of roughly 5 distinct failure modes observed (completeness-gap, citation-mismatch) — false-refusal, scenario-specific-reasoning, and retrieval-limited-comparison modes have zero independent representation.

**Overall: PARTIALLY FEASIBLE.** Judge development can proceed on this data. An independent holdout split is numerically constructible but would not currently support rigorous per-axis failure-detection validation — it would only support a coarse, PASS-heavy agreement check plus a one-family check each on Completeness and Citation-mismatch.

## 6. If proceeding: data gaps (no new questions created)

New validation examples must come from **new, independent question families** — not new phrasings of the 8 already-exposed families, and not the untouched confirmatory dataset (out of scope until separately approved).

| Axis | Gap | Approx. new independent families needed |
|---|---|---|
| Faithfulness | Zero unsupported-claim examples exist | 2–3 |
| Correctness | Only 1 (exposed) wrong-fact example exists | 2–3, distinct failure mechanism from q019's "range vs. specific value" |
| Task Completion | Both examples are same-mechanism (zero-hit false refusal) and exposed | 2–3, via a *different* mechanism (e.g. off-topic/evasive answer despite sufficient evidence) |
| Completeness | Only 1 independent family (q002) | 2–3 more, spanning different omission patterns |
| Citation Validity vs. Citation Support | The only failure (q013) fails both simultaneously — the two axes have never been empirically separated | New families isolating each: (a) citation to a non-retrieved section with otherwise-fine support, (b) citation to a real retrieved section that doesn't support its claim |
| Correct Abstention / Unsupported Answer | Zero fabrication-on-unanswerable examples exist | 2–3 unanswerable-question families where the system actually fabricates |

**Approximate total: 8–12 new independent question families**, none overlapping the 8 already-exposed families or their underlying policy/retrieval scenario.

## 7. Label / gold consistency check

- ✅ **q007 uses evaluation gold v1.1** (confirmed via `human_eval_metrics_v2.1.json`: `evaluation_gold_version: "v1.1"`; `generation_dataset_version` remains `"v1.0"`, i.e. generation was never rerun).
- ✅ **q007 Control and v1 Treatment were reconsidered under the revised gold**, and their existing Completeness=FAIL / GAS=FAIL labels **remain valid** — both still omit the retained B-or-better claim (`q007-c05`). Confirmed directly against `gold_revision_log.md` and cross-checked in `final_human_labels_v2.1.csv`.
- ✅ **q027's label correction is reflected consistently** — both Control and v1 Treatment show Completeness=PASS, Final GAS=PASS in the final CSV, matching the correction notes on response_ids R030/R040.

**Important distinction, stated explicitly per instruction:** the aggregate metric levels (Control GAS 68%, v1 Treatment GAS 72%, v2.1 Treatment GAS 80%) reflect **evaluation-instrument revisions in part, not purely model-performance changes.** The q007 gold correction and the q027 label correction were each applied identically to every arm they touch, so the **Control-vs-Treatment comparison itself is not distorted** by them — but the absolute percentages are not comparable to any hypothetical pre-correction baseline, and should not be read as evidence the underlying model or prompt changed.

## 8. Confirmation

**No split was frozen.** No `judge_calibration_split_v1.csv` (or equivalent) was created. No human label, gold file, or experiment result file was modified during this audit.
