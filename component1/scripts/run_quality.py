"""Run the signal-quality assessment on all STEW recordings.

Usage:  python scripts/run_quality.py [--config configs/quality.yaml]
Writes: results/quality/quality_windows.csv, quality_channels.csv, quality_summary.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from eegengine.config import config_path, load_config, resolve
from eegengine.io.stew import load_stew
from eegengine.quality.checks import QualityConfig
from eegengine.quality.report import build_quality_report, save_quality_report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quality.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    t0 = time.perf_counter()
    recs = load_stew(resolve(cfg["data"]["stew_dir"]), cache_path=resolve(cfg["data"]["cache"]))
    qcfg = QualityConfig.from_dict(cfg["quality"])
    windows, channels, summary = build_quality_report(recs, qcfg, cfg["dc_removal"]["highpass_hz"])

    out = resolve(cfg["output_dir"])
    save_quality_report(windows, channels, summary, out)
    shutil.copy(config_path(args.config), out / "config.yaml")

    print(f"Assessed {summary['n_recordings']} recordings, {summary['n_windows']} windows "
          f"({qcfg.window_s:g}s / step {qcfg.step_s:g}s) in {time.perf_counter() - t0:.1f}s")
    print("Window flags:", summary["flag_counts"], summary["flag_fractions"])
    print("By condition:", summary["flag_fractions_by_condition"])
    print("Windows failing each check:", summary["windows_failing_check"])
    top = list(summary["mean_channel_bad_fraction"].items())[:5]
    print("Channels most often bad:", ", ".join(f"{c} {v:.1%}" for c, v in top))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
