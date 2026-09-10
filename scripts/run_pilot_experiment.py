"""
Paired Control/Treatment pilot experiment (execution / data collection only).

Control:
    question -> retrieval -> current first-pass Generator.generate() -> final answer
Treatment:
    question -> SAME retrieval -> SAME first-pass draft -> one verifier pass
    -> KEEP / REVISE / ABSTAIN -> final answer

This script does NOT compute GAS, citation validity/support, evidence
coverage, or any judge/human-annotation score -- see heinzy/eval/experiment.py
and heinzy/generation/verify.py for what it does record (raw text, citations
as currently extracted, retrieval snapshot, usage, latency, provenance).
Scoring is a later, separate stage applied identically to both arms.

Run from repo root:
    # default: eval/questions_pilot_v1.0.jsonl, shared Chroma store, k from config
    python scripts/run_pilot_experiment.py

    # no Chroma reachable -> ingest into an in-process store
    python scripts/run_pilot_experiment.py --backend memory

    # reproducible question-order shuffle
    python scripts/run_pilot_experiment.py --shuffle --seed 7

    # verify against a different model than the first pass used
    python scripts/run_pilot_experiment.py --verifier-model llama3.2:latest
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.eval.experiment import (
    DEFAULT_EXECUTION_SEED,
    DatasetValidationError,
    build_report,
    ensure_tools_disabled,
    load_and_validate_rows,
    run_experiment,
    shuffled_order,
)
from heinzy.generation.generator import Generator
from heinzy.retrieval.retrieve import Retriever

DEFAULT_DATASET = Path("eval/questions_pilot_v1.0.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--backend", default=None, choices=["memory", "chroma"],
                     help="override vector_store.backend for this run only")
    ap.add_argument("--k", type=int, default=None, help="override retrieval k")
    ap.add_argument("--out", type=Path, default=None,
                     help="output JSON path (default: eval/results/pilot_experiment_<hash>_<ts>.json)")
    ap.add_argument("--shuffle", action="store_true",
                     help="run questions in a reproducible shuffled order instead of file order")
    ap.add_argument("--seed", type=int, default=DEFAULT_EXECUTION_SEED,
                     help="question execution order seed, used only with --shuffle "
                          "(arm order is never randomized: Control/shared draft always runs "
                          "before the Treatment verifier pass, since the draft is shared)")
    ap.add_argument("--verifier-model", default=None,
                     help="MODEL_TAG override for the verifier pass only; "
                          "defaults to the same model the first pass used")
    args = ap.parse_args()

    cfg = load_config()
    if args.backend is not None:
        cfg.vector_store.backend = args.backend

    # ---- 1-5: load + validate BEFORE any retrieval/model call ----
    try:
        rows = load_and_validate_rows(args.dataset)
    except DatasetValidationError as exc:
        print(f"ABORT: {args.dataset} failed validation ({len(exc.errors)} issue(s)):",
              file=sys.stderr)
        for e in exc.errors:
            print(f"  {e}", file=sys.stderr)
        return 2
    print(f"loaded {len(rows)} rows from {args.dataset} (all valid)")

    # ---- 7: build the handbook store ----
    # ingest_and_populate_store (heinzy/pipeline.py) only ever reads
    # data/corpus/*.pdf via heinzy/ingest/registry.py's register_corpus --
    # the QA gold JSONL under eval/ is never part of the ingested corpus.
    from heinzy.pipeline import ingest_and_populate_store

    store = ingest_and_populate_store(cfg)
    retriever = Retriever(cfg, store=store)
    generator = Generator(cfg)

    # ---- 6: verify tools are actually disabled, not just assumed off ----
    tools_were_enabled = ensure_tools_disabled(generator)
    if tools_were_enabled:
        print("=" * 72, file=sys.stderr)
        print("WARNING: tool-calling was enabled by the ambient config/environment", file=sys.stderr)
        print("(GOVERNANCE_SRC mount present and governance.enabled=true). It has", file=sys.stderr)
        print("been forcibly disabled for this experiment: Generator.use_tools is", file=sys.stderr)
        print("now False. The pilot's first pass must be a single, tool-free model", file=sys.stderr)
        print("call to be interpretable and comparable across arms.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
    print(f"tools_enabled=false (tools_originally_enabled={tools_were_enabled})")

    verifier_generator = generator
    if args.verifier_model:
        # Separate instance so the first pass and the verifier can run
        # different models; independently confirmed tool-free.
        verifier_generator = copy.copy(generator)
        verifier_generator.model_tag = args.verifier_model
        ensure_tools_disabled(verifier_generator)

    # ---- 8: optional reproducible question-order shuffle ----
    execution_order = None
    execution_seed = None
    if args.shuffle:
        execution_order = shuffled_order(len(rows), seed=args.seed)
        execution_seed = args.seed

    # ---- 9: run ----
    def _progress(result) -> None:
        print(f"  {result.question_id}: control_refused={result.control.refused} "
              f"verifier={result.treatment.verifier_decision}")

    t0 = time.time()
    results = run_experiment(
        rows, retriever, generator,
        verifier_generator=verifier_generator,
        k=args.k,
        execution_order=execution_order,
        on_result=_progress,
    )
    elapsed = round(time.time() - t0, 2)

    report = build_report(
        results,
        cfg=cfg,
        generator=generator,
        verifier_generator=verifier_generator,
        dataset_path=str(args.dataset),
        dataset_row_count=len(rows),
        k=args.k if args.k is not None else cfg.retrieval.k,
        backend=cfg.vector_store.backend,
        tools_originally_enabled=tools_were_enabled,
        execution_seed=execution_seed,
    )

    # ---- 10: write structured JSON result ----
    out_path = args.out or (DEFAULT_OUT_DIR / f"pilot_experiment_{cfg.config_hash}_{report.timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2))

    # ---- 11: concise summary. No GAS, no citation validity, no scoring. ----
    decisions = [r.treatment.verifier_decision for r in results]
    n_keep = decisions.count("KEEP")
    n_revise = decisions.count("REVISE")
    n_abstain = decisions.count("ABSTAIN")
    n_zero_hit_guard = sum(1 for r in results if r.treatment.zero_hit_guard_triggered)

    control_e2e = [r.control_e2e_latency_seconds for r in results]
    treatment_e2e = [r.treatment.effective_e2e_latency_seconds for r in results]
    total_tokens = [r.treatment.effective_total_usage.total_tokens for r in results]

    print("\n=== Heinzy pilot experiment (execution only, no scoring) ===")
    print(f"questions run    : {len(results)}")
    print(f"elapsed          : {elapsed}s")
    print(f"output           : {out_path}")
    print(f"generator model  : {report.generator_model}")
    print(f"verifier model   : {report.verifier_model}")
    print(f"config_hash      : {report.config_hash}")
    print(f"verifier prompt  : {report.verifier_prompt_version} ({report.verifier_prompt_fingerprint})")
    print(f"treatment strategy: {report.treatment_strategy_version}")
    print(f"execution_seed   : {report.execution_seed} (shuffle={'on' if args.shuffle else 'off'})")
    print(f"decisions        : KEEP={n_keep} REVISE={n_revise} ABSTAIN={n_abstain} "
          f"(zero_hit_guard_triggered={n_zero_hit_guard})")
    print(f"control e2e lat  : mean={_mean(control_e2e)}s median={_median(control_e2e)}s")
    print(f"treatment e2e lat: mean={_mean(treatment_e2e)}s median={_median(treatment_e2e)}s")
    print(f"treatment tokens : mean={_mean(total_tokens)} median={_median(total_tokens)} "
          "(None entries excluded -- provider did not report usage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
