"""Run the whole pipeline from the raw STEW files, in order, with one command.

Usage:
  python scripts/run_all.py                 # full pipeline (several hours on a 4-core laptop CPU)
  python scripts/run_all.py --quick         # everything except the long LOSO/deep runs (~10 min):
                                            #   uses existing results for those steps if present
  python scripts/run_all.py --from deep     # resume from a step
Every step is an ordinary script, so any step can also be run on its own. Each step's output is
logged to results/logs/run_all_<step>.log; the run stops at the first failing step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from eegengine.config import PROJECT_ROOT, RESULTS_DIR

PY = sys.executable
STEPS: list[tuple[str, list[str], bool]] = [   # (name, command, long-running?)
    ("eda", [PY, "scripts/explore_dataset.py"], False),
    ("quality", [PY, "scripts/run_quality.py"], False),
    ("preprocess", [PY, "scripts/run_preprocess.py", "--ica-demo"], False),
    ("preprocess_2s", [PY, "scripts/run_preprocess.py", "--config", "preprocess_2s.yaml"], False),
    ("features", [PY, "scripts/run_features.py"], False),
    ("baselines", [PY, "scripts/run_baselines.py", "--skip-existing"], True),
    ("baselines_rest", [PY, "scripts/run_baselines.py", "--config", "experiments/classical_rest.yaml", "--skip-existing"], True),
    ("ablation", [PY, "scripts/run_ablation.py", "--skip-existing"], True),
    ("classical_kfold5", [PY, "scripts/run_baselines.py", "--config", "experiments/classical_kfold5.yaml", "--skip-existing"], True),
    ("deep_smoke", [PY, "scripts/run_deep.py", "--smoke"], False),
    ("deep", [PY, "scripts/run_deep.py", "--skip-existing"], True),
    ("deep_stats", [PY, "scripts/deep_stats.py"], False),
    ("final_classical", [PY, "scripts/train_final_classical.py"], False),
    ("final_deep", [PY, "scripts/run_deep.py", "--final", "A10_hybrid"], True),
    ("report", [PY, "scripts/export_report.py"], False),
    ("contract", [PY, "scripts/export_contract.py"], False),
    ("audit", [PY, "scripts/audit_leakage.py"], True),
    ("demo", [PY, "scripts/export_demo.py"], False),
    ("tests", [PY, "-m", "pytest"], False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip long-running steps")
    ap.add_argument("--from", dest="start", help="first step to run")
    args = ap.parse_args()
    logs = RESULTS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    names = [s[0] for s in STEPS]
    first = names.index(args.start) if args.start else 0
    for name, cmd, long in STEPS[first:]:
        if args.quick and long:
            print(f"[skip] {name} (long-running; --quick)")
            continue
        t = time.perf_counter()
        print(f"[run ] {name}: {' '.join(cmd[1:])}", flush=True)
        with open(logs / f"run_all_{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"[{'ok' if rc == 0 else 'FAIL'}  ] {name} ({time.perf_counter() - t:.0f} s)", flush=True)
        if rc != 0:
            print(f"Step {name} failed — see results/logs/run_all_{name}.log")
            sys.exit(rc)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
