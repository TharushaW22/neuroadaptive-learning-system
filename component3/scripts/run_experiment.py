"""Step 8: run the experiment (explanations + per-instance metrics), then the analysis.

  EEG pilot (< 30 min):  .venv/bin/python scripts/run_experiment.py --pilot
  EEG full run:          .venv/bin/python scripts/run_experiment.py
  Vision pilot:          .venv/bin/python scripts/run_experiment.py --vision --pilot   (results/vision_pilot)
  Vision full run:       .venv/bin/python scripts/run_experiment.py --vision            (results/vision_full)
Options: --run-name NAME, --workers N, --limit N (first N instances per context, for smoke tests),
--stage explain|analyse|all. Re-running continues where it stopped.
--vision uses configs/run_vision.yaml and refuses to start unless scripts/check_vision_ready.py passes.
"""
import argparse
import sys
import time

import _bootstrap  # noqa: F401

from xaicomp.config import RUN_FILES, load_c3_config, load_context, run_settings
from xaicomp.paths import RESULTS_DIR
from xaicomp.run.orchestrator import default_run_name, run_experiment


def vision_gate(contexts: list[str]) -> bool:
    """True when every vision context passes the readiness check; prints the checklist otherwise."""
    from xaicomp.run.vision_readiness import check_vision_ready, format_report

    ok = True
    for name in contexts:
        ctx = load_context(name)
        if ctx.get("modality") != "vision" or ctx.get("negative_control"):
            continue
        res = check_vision_ready(ctx, compute_accuracy=False)
        if not res["ready"]:
            print(format_report(res))
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--vision", action="store_true", help="run the vision context (configs/run_vision.yaml)")
    ap.add_argument("--run-name")
    ap.add_argument("--contexts", nargs="*")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stage", choices=["explain", "analyse", "all"], default="all")
    args = ap.parse_args()
    run_file = RUN_FILES["vision" if args.vision else "eeg"]
    contexts = args.contexts or load_c3_config(run_file)["contexts"]
    if not args.vision and any(load_context(c).get("modality") == "vision" for c in contexts):
        print("REFUSED: vision contexts need --vision (they use configs/run_vision.yaml).")
        return 2
    if args.vision and args.stage in ("explain", "all") and not vision_gate(contexts):
        print("\nREFUSED: the vision context cannot run yet — Component 2's trained model and features are needed "
              "(see the checklist above and docs/VISION_RUNBOOK.md).")
        return 2
    t0 = time.perf_counter()
    run_root = RESULTS_DIR / (args.run_name or default_run_name(run_settings(args.pilot, run_file)))
    if args.stage in ("explain", "all"):
        run_root = run_experiment(args.pilot, args.run_name, args.contexts, args.workers, args.limit, run_file=run_file)
    if args.stage in ("analyse", "all"):
        from xaicomp.run.analysis import analyse_run
        analyse_run(run_root, pilot=args.pilot, run_file=run_file)
    print(f"finished in {(time.perf_counter() - t0) / 60:.1f} min -> {run_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
