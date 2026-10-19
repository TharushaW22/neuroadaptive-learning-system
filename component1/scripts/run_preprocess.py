"""Phase 2: preprocess + epoch all STEW recordings.

Usage:  python scripts/run_preprocess.py [--config preprocess.yaml] [--subjects 1 2] [--ica-demo]
Writes:
  data/processed/epochs_<hash>.npz          X (n_epochs, 14, win) float32
  data/processed/epochs_<hash>_meta.csv     one row per epoch (+ .parquet)
  results/preprocessing/                    logs, quality comparison, usable epochs, PSD plots
  docs/labels.md                            label definitions + class counts (generated)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from eegengine.config import PROJECT_ROOT, config_path, load_config, resolve, set_seed
from eegengine.io.stew import CHANNELS, load_stew
from eegengine.labels import TASKS, add_labels, label_counts
from eegengine.preprocess.dc import remove_dc
from eegengine.preprocess.filters import filter_recording
from eegengine.preprocess.ica import run_ica
from eegengine.preprocess.normalize import normalize_epochs
from eegengine.preprocess.pipeline import config_hash, epoch_quality_config, preprocess_recording
from eegengine.quality.checks import assess_recording

FLAGS = ("good", "warn", "bad")


def flag_rates(flags: list[str]) -> dict:
    n = len(flags)
    return {f: round(sum(x == f for x in flags) / n, 4) for f in FLAGS}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="preprocess.yaml")
    ap.add_argument("--subjects", type=int, nargs="*", help="subset for a smoke run")
    ap.add_argument("--ica-demo", action="store_true", help="run ICA on the plot subjects and log it")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    h = config_hash(cfg)
    out_dir = resolve(cfg["plots"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    recs = load_stew(resolve(cfg["data"]["stew_dir"]), subjects=args.subjects,
                     cache_path=resolve(cfg["data"]["cache"]))
    qcfg = epoch_quality_config(cfg)
    fs = recs[0].fs

    # --- raw vs filtered quality, and an amplitude-threshold sweep on the filtered signal
    raw_flags, filt_flags = [], []
    sweep = {thr: [] for thr in (100.0, 120.0, 150.0, 200.0)}
    filtered_cache = {}
    for r in recs:
        raw_rows, _ = assess_recording(remove_dc(r.data, fs, 0.5), fs, CHANNELS, qcfg)
        raw_flags += [q["flag"] for q in raw_rows]
        f = filter_recording(r.data, fs, cfg["filter"])
        filtered_cache[r.key] = f
        rows, _ = assess_recording(f, fs, CHANNELS, qcfg)
        filt_flags += [q["flag"] for q in rows]
        for thr in sweep:
            q2 = epoch_quality_config({**cfg, "quality": {**cfg["quality"], "max_abs_uv": thr}})
            sweep[thr] += [q["flag"] for q in assess_recording(f, fs, CHANNELS, q2)[0]]

    # --- full pipeline per recording
    X, meta, interp_log, q_final = [], [], [], []
    for r in recs:
        res = preprocess_recording(r, cfg)
        X.append(res["epochs"])
        meta += res["meta"]
        q_final += [q["flag"] for q in res["quality_rows"]]
        bl = res["bad_channel_log"]
        if bl["detected"]:
            interp_log.append({"recording": r.key, **bl})
    X = np.concatenate(X)
    meta = add_labels(pd.DataFrame(meta))
    X = normalize_epochs(X, meta, cfg["normalisation"]).astype(np.float32)
    assert np.isfinite(X).all(), "NaN/Inf in epochs"

    # --- save epochs + metadata
    data_dir = resolve(cfg["data"]["out_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = h if not args.subjects else f"{h}_smoke"
    np.savez(data_dir / f"epochs_{tag}.npz", X=X)
    meta.to_csv(data_dir / f"epochs_{tag}_meta.csv", index=False)
    meta.to_parquet(data_dir / f"epochs_{tag}_meta.parquet", index=False)
    with open(data_dir / f"epochs_{tag}_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    if args.subjects:  # smoke run: no reports
        print(f"[smoke] {len(recs)} recordings -> X {X.shape}, saved epochs_{tag}.npz "
              f"in {time.perf_counter() - t0:.1f}s")
        return

    # --- reports
    comparison = {
        "n_windows": len(raw_flags),
        "raw_dc_removed": flag_rates(raw_flags),
        "filtered": flag_rates(filt_flags),
        "filtered_interpolated_final": flag_rates(q_final),
        "amplitude_threshold_sweep_filtered": {f"{t:g}": flag_rates(v) for t, v in sweep.items()},
        "phase1_raw_report_rules": "see results/quality/quality_summary.json (no blink tolerance)",
        "quality_config": qcfg.to_dict(),
    }
    with open(out_dir / "quality_raw_vs_filtered.json", "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)
    with open(out_dir / "interpolation_log.json", "w", encoding="utf-8") as fh:
        json.dump(interp_log, fh, indent=2)
    pd.DataFrame([{"recording": d["recording"], "channel": c["channel"], "reasons": "+".join(c["reasons"]),
                   "bad_fraction": c["bad_fraction"], "std_ratio": c["std_ratio"],
                   "interpolated": c["channel"] in d["interpolated"]}
                  for d in interp_log for c in d["detected"]]).to_csv(out_dir / "interpolation_log.csv", index=False)

    usable = (meta.assign(usable=meta.quality_flag != "bad")
              .groupby(["subject_id", "condition"]).usable.mean().unstack().rename(columns=lambda c: f"usable_{c}"))
    usable["usable_all"] = meta.assign(usable=meta.quality_flag != "bad").groupby("subject_id").usable.mean()
    usable = usable.round(3)
    usable.to_csv(out_dir / "usable_epochs_per_subject.csv")
    low_usable = usable[(usable[["usable_lo", "usable_hi"]] < 0.5).any(axis=1)]

    counts_epochs = label_counts(meta, unit="epochs")
    rec_meta = meta.drop_duplicates(["subject_id", "condition"])
    counts_recs = label_counts(rec_meta, unit="recordings")
    with open(out_dir / "label_counts.json", "w", encoding="utf-8") as fh:
        json.dump({"epochs": counts_epochs, "recordings": counts_recs}, fh, indent=2)
    write_labels_doc(counts_recs, counts_epochs, PROJECT_ROOT / "docs" / "labels.md")

    summary = {
        "config_hash": h,
        "epochs_file": f"data/processed/epochs_{h}.npz",
        "X_shape": list(X.shape),
        "epochs_per_recording": int(meta.groupby(["subject_id", "condition"]).size().iloc[0]),
        "normalisation": cfg["normalisation"],
        "n_recordings_with_interpolation": sum(bool(d["interpolated"]) for d in interp_log),
        "n_channels_interpolated": sum(len(d["interpolated"]) for d in interp_log),
        "n_channels_detected_not_interpolated": sum(len(d["not_interpolated"]) for d in interp_log),
        "final_flag_rates": flag_rates(q_final),
        "usable_fraction_overall": round(float((meta.quality_flag != "bad").mean()), 4),
        "recordings_below_50pct_usable": [
            f"sub{s:02d}_{c}" for s, row in low_usable.iterrows() for c in ("lo", "hi") if row[f"usable_{c}"] < 0.5],
        "note_low_usable": "These subjects are KEPT. Bad windows are only excluded when an experiment "
                           "sets exclude_bad_windows=true.",
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    with open(out_dir / "preprocess_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    shutil.copy(config_path(args.config), out_dir / "config.yaml")

    for s in cfg["plots"]["psd_subjects"]:
        plot_psd_before_after(recs, filtered_cache, X, meta, s, fs, out_dir)

    if args.ica_demo:
        ica_logs = {}
        for s in cfg["plots"]["psd_subjects"]:
            for r in (r for r in recs if r.subject_id == s):
                _, log = run_ica(filtered_cache[r.key], fs, CHANNELS, {**cfg["ica"], "enabled": True})
                ica_logs[r.key] = log
        with open(out_dir / "ica_demo_log.json", "w", encoding="utf-8") as fh:
            json.dump(ica_logs, fh, indent=2)
        print("ICA demo:", ica_logs)

    print(json.dumps(summary, indent=2))
    print("Quality raw vs filtered:", json.dumps({k: comparison[k] for k in
          ("raw_dc_removed", "filtered", "filtered_interpolated_final", "amplitude_threshold_sweep_filtered")}, indent=1))


def plot_psd_before_after(recs, filtered_cache, X, meta, subject, fs, out_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, cond in zip(axes, ("lo", "hi")):
        r = next(r for r in recs if r.subject_id == subject and r.condition == cond)
        before = remove_dc(r.data, fs, highpass_hz=None)  # mean removal only = "raw"
        idx = np.where((meta.subject_id == subject) & (meta.condition == cond))[0]
        f, p_b = welch(before, fs=fs, nperseg=2 * fs, axis=1)
        _, p_a = welch(X[idx], fs=fs, nperseg=2 * fs, axis=-1)
        keep = (f >= 0.25) & (f <= 64)
        ax.plot(f[keep], 10 * np.log10(p_b.mean(0)[keep]), color="#2a78d6", label="before (DC removed)")
        ax.plot(f[keep], 10 * np.log10(p_a.mean((0, 1))[keep]), color="#eb6834", label="after (epochs)")
        for edge in (0.5, 45):
            ax.axvline(edge, color="#a3a29d", lw=0.8, ls="--")
        ax.set_title(f"Subject {subject:02d} - {'rest' if cond == 'lo' else 'task'}", loc="left")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Power (dB re 1 uV²/Hz), mean of 14 ch")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("PSD before vs after preprocessing (0.5-45 Hz FIR, 50 Hz notch, interpolation)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"psd_before_after_sub{subject:02d}.png", dpi=150)
    plt.close(fig)


def write_labels_doc(rec_counts: dict, ep_counts: dict, path) -> None:
    lines = [
        "# Labels (generated by scripts/run_preprocess.py — do not edit by hand)",
        "",
        "STEW only provides **workload** information: the condition (rest `lo` / SIMKAP task `hi`) and a",
        "self-reported workload rating (1–9) per recording. Subjects 5, 24 and 42 have no rating.",
        "Masked labels are stored as `-1`; subjects are never dropped — each task uses its mask.",
        "",
        "| Task | Column | Definition | Subjects |",
        "|---|---|---|---|",
        f"| T1 workload_binary | `y_binary` | lo = 0 (low), hi = 1 (high) | {rec_counts['workload_binary']['n_subjects']} |",
        f"| T2 workload_level | `y_level` | own rating 1–3 low, 4–6 medium, 7–9 high; unrated masked | {rec_counts['workload_level']['n_subjects']} |",
        f"| T3 workload_within_task | `y_within` | hi recordings only: rating 4–6 moderate, 7–9 high; lo + unrated masked | {rec_counts['workload_within_task']['n_subjects']} |",
        "",
        "## Class counts",
        "",
        "| Task | Unit | " + " | ".join(["class counts", "by condition (lo / hi)"]) + " |",
        "|---|---|---|---|",
    ]
    for counts in (rec_counts, ep_counts):
        for task, c in counts.items():
            cls = ", ".join(f"{k} {v}" for k, v in c["classes"].items())
            bc = "; ".join(f"{cond}: " + "/".join(str(v) for v in c["by_condition"][cond].values())
                           for cond in ("lo", "hi"))
            lines.append(f"| {TASKS[task]['short']} {task} | {c['unit']} | {cls} | {bc} |")
    t2 = rec_counts["workload_level"]["by_condition"]
    lines += [
        "",
        "## Why T2 is confounded and T3 matters",
        "",
        f"For T2, rest recordings are low/medium/high = {'/'.join(str(v) for v in t2['lo'].values())} "
        f"and task recordings are {'/'.join(str(v) for v in t2['hi'].values())}. "
        "So a T2 classifier can score well just by detecting rest vs task (T1) — it says little about "
        "*graded* workload.",
        "",
        "T3 uses only the task recordings, so both classes share the same condition, the same task "
        "and the same recording setup; only the self-reported workload differs. It is the "
        "confound-free workload task and is treated as a primary result, even though it has only "
        f"{rec_counts['workload_within_task']['n']} labelled recordings (one per rated subject), so its "
        "estimates are noisy. Because every subject contributes exactly one class in T3, per-subject "
        "metrics such as macro-F1 are undefined; T3 is evaluated on pooled out-of-fold predictions and "
        "on one aggregated prediction per subject.",
        "",
        "Attention, fatigue and learning readiness have **no labels** in STEW. Their model heads exist "
        "but are masked off; the dashboard shows heuristic EEG indices for them instead.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
