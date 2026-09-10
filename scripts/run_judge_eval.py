"""
Judge v1 CLI (Heinzy RAG evaluation) -- grades ALREADY-SAVED experiment
responses against gold v1.2 using the Judge v1 prompt/rubric/schema. Never
runs retrieval, generation, or the Treatment verifier; never regenerates a
response.

This is EXECUTION only (see heinzy/eval/judge.py module docstring for the
Judge-execution / calibration-comparison boundary): saved response + gold +
retrieved evidence -> Judge result. It never loads, compares against, or
even imports the human-label file
(eval/annotation/final_human_labels_v2.1.csv) -- that comparison is a
separate, later step.

Examples (see also the final report for exact future commands):

    # dry run: build everything up to the rendered Judge input, make ZERO
    # model calls, and print a summary.
    python scripts/run_judge_eval.py \\
        --experiment-result eval/results/pilot_full_30q_v2.1_20260910T011413Z.json \\
        --gold eval/questions_pilot_v1.2.jsonl \\
        --dry-run

    # inspect exactly one rendered blinded Judge request (no model call)
    python scripts/run_judge_eval.py \\
        --experiment-result eval/results/pilot_full_30q_v2.1_20260910T011413Z.json \\
        --question-id q001 --inspect

    # real run (NOT executed by this task) -- requires --judge-model
    python scripts/run_judge_eval.py \\
        --experiment-result eval/results/pilot_full_30q_v2.1_20260910T011413Z.json \\
        --judge-provider ollama --judge-model llama3.2:latest \\
        --require-full-coverage \\
        --output eval/results/judge_v1_dev_60resp_<timestamp>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from heinzy.eval.dataset import DatasetError, load_dataset, validate_dataset
from heinzy.eval.judge import (
    ARM_CONTROL,
    ARM_TREATMENT,
    DEFAULT_ARMS,
    HTTPJudgeClient,
    JudgeInputError,
    ModelConfig,
    assert_control_sources_consistent,
    assert_unique_flattened,
    build_evaluation_subject,
    build_gold_index,
    compare_control_sources,
    compute_design_provenance,
    evaluate_subject,
    flatten_experiment_report,
    join_gold,
    load_judge_prompt_variants,
    load_output_schema,
    render_judge_request,
    validate_full_coverage,
)

DEFAULT_GOLD = Path("eval/questions_pilot_v1.2.jsonl")
DEFAULT_PROMPT = Path("eval/judge/judge_prompt_v1.txt")
DEFAULT_RUBRIC = Path("eval/judge/judge_rubric_v1.md")
DEFAULT_SCHEMA = Path("eval/judge/judge_output_schema_v1.json")
DEFAULT_OUT_DIR = Path("eval/results")


def _load_and_validate_gold(path: Path):
    try:
        rows = load_dataset(path)
    except DatasetError as exc:
        print(f"ABORT: {path} failed to load: {exc}", file=sys.stderr)
        sys.exit(2)
    errors = validate_dataset(rows)
    if errors:
        print(f"ABORT: {path} failed validation ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(2)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment-result", type=Path, required=True,
                     help="saved experiment-result JSON (e.g. pilot_full_30q_v2.1_*.json); "
                          "flattened for both Control and Treatment by default")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--output", type=Path, default=None,
                     help="output JSON path (default: eval/results/judge_v1_<ts>.json); "
                          "not written at all in --dry-run/--inspect unless given explicitly")
    ap.add_argument("--judge-provider", default="ollama")
    ap.add_argument("--judge-model", default=None,
                     help="required for a real run; not required for --dry-run/--inspect")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None,
                     help="grade only the first N flattened subjects (after --question-id filtering)")
    ap.add_argument("--question-id", action="append", default=None,
                     help="restrict to this question_id; repeatable for a smoke-test mix")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=[ARM_CONTROL, ARM_TREATMENT])
    ap.add_argument("--dry-run", action="store_true",
                     help="build everything through the rendered Judge input and fingerprints; "
                          "make ZERO model calls")
    ap.add_argument("--inspect", action="store_true",
                     help="print exactly one rendered blinded Judge request and exit; implies --dry-run; "
                          "combine with --question-id to pick which one")
    ap.add_argument("--control-consistency-check", type=Path, default=None,
                     help="another saved experiment-result JSON to compare Control against; "
                          "aborts on any mismatch instead of proceeding")
    ap.add_argument("--require-full-coverage", action="store_true",
                     help="enforce exactly one response per gold question_id per requested arm "
                          "(the full 30x2 development-run shape) before doing anything else")
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--evaluation-gold-version", default="v1.2")
    args = ap.parse_args()

    if args.inspect:
        args.dry_run = True

    report = json.loads(args.experiment_result.read_text(encoding="utf-8"))

    if args.control_consistency_check is not None:
        other = json.loads(args.control_consistency_check.read_text(encoding="utf-8"))
        comparisons = compare_control_sources(report, other)
        try:
            assert_control_sources_consistent(comparisons)
        except JudgeInputError as exc:
            print("ABORT: Control-source consistency check failed:", file=sys.stderr)
            for e in exc.errors:
                print(f"  {e}", file=sys.stderr)
            return 2
        print(f"Control consistency check: {len(comparisons)} question(s) compared, all match.")

    rows = _load_and_validate_gold(args.gold)
    print(f"loaded {len(rows)} gold rows from {args.gold} (all valid)")
    gold_index = build_gold_index(rows)

    all_flat = flatten_experiment_report(
        report, str(args.experiment_result), arms=tuple(args.arms)
    )
    try:
        assert_unique_flattened(all_flat)
    except JudgeInputError as exc:
        print("ABORT: response identity/uniqueness check failed:", file=sys.stderr)
        for e in exc.errors:
            print(f"  {e}", file=sys.stderr)
        return 2

    if args.require_full_coverage:
        try:
            validate_full_coverage(all_flat, rows, expected_arms=tuple(args.arms))
        except JudgeInputError as exc:
            print("ABORT: full-coverage check failed:", file=sys.stderr)
            for e in exc.errors:
                print(f"  {e}", file=sys.stderr)
            return 2
        print(f"full-coverage check OK: {len(rows)} question(s) x {len(args.arms)} arm(s)")

    flat = all_flat
    if args.question_id:
        wanted = set(args.question_id)
        flat = [f for f in flat if f.question_id in wanted]
        missing = wanted - {f.question_id for f in flat}
        if missing:
            print(f"ABORT: --question-id not found in experiment result: {sorted(missing)}",
                  file=sys.stderr)
            return 2
    if args.limit is not None:
        flat = flat[: args.limit]

    if not flat:
        print("ABORT: no evaluation subjects selected.", file=sys.stderr)
        return 2

    subjects = []
    for f in flat:
        try:
            gold_row = join_gold(f, gold_index)
            subject = build_evaluation_subject(f, gold_row)
        except JudgeInputError as exc:
            print("ABORT: gold join failed:", file=sys.stderr)
            for e in exc.errors:
                print(f"  {e}", file=sys.stderr)
            return 2
        subjects.append((f, subject))

    variants = load_judge_prompt_variants(args.prompt)
    schema_doc = load_output_schema(args.schema)
    design_provenance = compute_design_provenance(
        prompt_path=args.prompt, rubric_path=args.rubric, schema_path=args.schema,
        gold_path=args.gold, evaluation_gold_version=args.evaluation_gold_version,
    )

    if args.inspect:
        f, subject = subjects[0]
        rendered = render_judge_request(subject, variants)
        print("=" * 78)
        print(f"INSPECTION ONLY -- NOT A JUDGE RESULT. question_id={f.question_id} arm={f.arm}")
        print("No model call was made.")
        print("=" * 78)
        print("\n--- SYSTEM MESSAGE ---\n")
        print(rendered.system)
        print("\n--- USER MESSAGE ---\n")
        print(rendered.user)
        return 0

    if args.dry_run:
        print("=" * 78)
        print(f"DRY RUN -- NOT JUDGE GRADING RESULTS. {len(subjects)} subject(s) prepared, "
              "ZERO model calls made.")
        print("=" * 78)
        inspection = []
        for f, subject in subjects:
            rendered = render_judge_request(subject, variants)
            from heinzy.eval.judge import (
                evaluation_subject_fingerprint,
                fingerprint_retrieved_context,
                sha256_text,
            )
            inspection.append(
                {
                    "question_id": f.question_id,
                    "arm": f.arm,
                    "answerable": subject.answerable,
                    "generated_response_fingerprint": sha256_text(subject.generated_response),
                    "retrieved_context_fingerprint": fingerprint_retrieved_context(
                        subject.retrieved_chunks
                    ),
                    "evaluation_subject_fingerprint": evaluation_subject_fingerprint(subject),
                    "rendered_judge_input_fingerprint": sha256_text(
                        rendered.system + "\x1e" + rendered.user
                    ),
                    "citation_count": len(subject.citation_resolutions),
                }
            )
            print(f"  {f.question_id} [{f.arm}]: answerable={subject.answerable} "
                  f"citations={len(subject.citation_resolutions)}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(
                {"dry_run": True, "note": "NOT JUDGE GRADING RESULTS", "subjects": inspection},
                indent=2,
            ))
            print(f"dry-run inspection output: {args.output}")
        return 0

    # ---- real run ----
    if not args.judge_model:
        print("ABORT: --judge-model is required for a real run.", file=sys.stderr)
        return 2

    client = HTTPJudgeClient(provider=args.judge_provider, model=args.judge_model)
    model_config = ModelConfig(
        judge_provider=args.judge_provider, judge_model=args.judge_model,
        temperature=args.temperature, timeout=args.timeout,
    )

    results = []
    for f, subject in subjects:
        result = evaluate_subject(
            subject, variants, client, model_config=model_config,
            design_provenance=design_provenance, schema_doc=schema_doc,
            source_experiment_file=f.source_experiment_file,
            source_record_index=f.source_record_index,
            arm=f.arm, response_id=f.response_id,
        )
        results.append(result)
        print(f"  {f.question_id} [{f.arm}]: {result.execution_status}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or (DEFAULT_OUT_DIR / f"judge_v1_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "source_experiment_file": str(args.experiment_result),
        "evaluation_gold_version": args.evaluation_gold_version,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "temperature": args.temperature,
        "results": [dataclasses.asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\noutput: {out_path}")
    print(f"{len(results)} subject(s) graded "
          f"(SUCCESS={sum(1 for r in results if r.execution_status == 'SUCCESS')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
