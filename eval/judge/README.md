# Heinzy LLM-as-a-Judge

Heinzy is a governed, retrieval-grounded RAG assistant. This module evaluates
response quality against human-reviewed references across:

- Task Completion
- Correctness
- Completeness
- Faithfulness
- Citation Support
- Correct Abstention
- Unsupported Answer

Citation Validity is evaluated separately through deterministic citation
resolution logic rather than by the LLM Judge.

## Final selected configuration

- **Judge:** `gpt-oss:20b`
- **Prompt:** `judge_prompt_v1.2.txt`
- **Rubric:** `judge_rubric_v1.2.md`
- **Schema:** `judge_output_schema_v1.json`
- **Temperature:** `0`
- **Reasoning:** `medium`
- **Development gold:** `questions_pilot_v1.4.jsonl`
- **Human labels:** `final_human_labels_v2.2.csv`

## Development calibration result

- Initial baseline: **251/270 = 92.96%** Human–Judge semantic-axis agreement
- Final development result: **267/270 = 98.89%**
- Mismatches reduced from **19 to 3**

**98.89% is development-set Human–Judge agreement after calibration and
adjudication. It is not independent validation accuracy.**

## Remaining known disagreements

1. **q002 Treatment — Completeness:** Human FAIL / Judge PASS. The response
   omits the specific 162-unit value, but the Judge sometimes treats the
   generic minimum-unit statement as sufficient.

2. **q007 Treatment — Completeness:** Human FAIL / Judge PASS. The answer
   mentions "B or better" only in a QPA context; the Judge fails to distinguish
   that from the required condition that the grade is necessary for units to
   count toward the degree.

3. **q019 Treatment — Correctness:** Human PASS / Judge FAIL. GPT-OSS
   consistently conflates two different quantities: 18 remaining units needed
   and a 6-unit difference between course options. Repeated GPT-OSS runs
   reproduced this failure.

## Calibration decision

An additional calibration candidate was tested to address the q019 numeric
reasoning failure. It did not fix q019 and introduced cross-axis regressions,
reducing agreement from **266/270 to 261/270** under the same comparison setup.
The final selected Judge therefore remained **v1.2**.

## Next stage

The frozen Judge will be evaluated on independent validation data using new
question families:

- **Natural-response validation** — ordinary system responses on new questions.
- **Targeted challenge validation** — deliberately difficult examples covering
  numeric confusion, missing conditions, unsupported claims, citation edge cases,
  and abstention behavior.

The current 30-question development set will not be used for further Judge tuning.
