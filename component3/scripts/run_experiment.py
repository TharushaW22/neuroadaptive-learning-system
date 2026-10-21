"""Step 8: run the experiment (explanations + per-instance metrics), then the analysis.

  Pilot (< 30 min):  .venv/bin/python scripts/run_experiment.py --pilot
  Full run:          .venv/bin/python scripts/run_experiment.py
Options: --run-name NAME (default pilot / full), --workers N, --limit N (first N instances per
context, for smoke tests), --stage explain|analyse|all. Re-running continues where it stopped.
"""
import argparse
import time

import _bootstrap  # noqa: F401

from xaicomp.paths import RESULTS_DIR
from xaicomp.run.orchestrator import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--run-name")
    ap.add_argument("--contexts", nargs="*")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stage", choices=["explain", "analyse", "all"], default="all")
    args = ap.parse_args()
    t0 = time.perf_counter()
    run_root = RESULTS_DIR / (args.run_name or ("pilot" if args.pilot else "full"))
    if args.stage in ("explain", "all"):
        run_root = run_experiment(args.pilot, args.run_name, args.contexts, args.workers, args.limit)
    if args.stage in ("analyse", "all"):
        from xaicomp.run.analysis import analyse_run
        analyse_run(run_root, pilot=args.pilot)
    print(f"finished in {(time.perf_counter() - t0) / 60:.1f} min -> {run_root}")


if __name__ == "__main__":
    main()
