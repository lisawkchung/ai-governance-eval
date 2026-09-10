# Heinzy LLM-as-a-Judge — Design v1

**Status: DESIGN ONLY. No judge has ever been run. No dev/holdout split
exists.**

| File | Purpose |
|---|---|
| `judge_rubric_v1.md` | Master human-readable rubric: axis definitions, GAS composition, human-label/rubric conflicts found during design, ambiguous decisions flagged for review. |
| `judge_prompt_v1.txt` | The actual LLM-facing prompt text (Answerable and Unanswerable variants), never yet sent to a model. |
| `judge_output_schema_v1.json` | Strict JSON Schema for the judge's output, plus documentation of the downstream (non-LLM) Citation Validity and Final GAS fields. |
| `judge_calibration_plan_v1.md` | How this prompt will be calibrated (development-only, on the existing pilot) and later independently validated (natural-response + targeted-challenge validation on new question families, not yet created). |

Read `judge_rubric_v1.md` §0 and `judge_calibration_plan_v1.md` §1 before
citing any agreement number from the current 30-question pilot — it is
judge-development data only, with little to no independent failure-example
coverage on most axes (full detail:
`eval/annotation/judge_split_feasibility_audit_v1.md`).
