"""Exploratory summary of the STEW dataset.

Usage:  python scripts/explore_dataset.py
Prints a dataset summary, the rating distribution and lo-vs-hi rating statistics, and saves
plots + dataset_summary.json to results/eda/.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon

from eegengine.config import RESULTS_DIR, resolve
from eegengine.io.stew import CHANNELS, LEVEL_NAMES, load_stew, recordings_to_frame
from eegengine.preprocess.dc import remove_dc

OUT = RESULTS_DIR / "eda"
COLORS = {"lo": "#2a78d6", "hi": "#eb6834"}          # rest = blue, task = orange
NAMES = {"lo": "Rest (lo)", "hi": "SIMKAP task (hi)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 1.5,
})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_stew(cache_path=resolve("data/processed/stew_raw_cache.npz"))
    meta = recordings_to_frame(recs)

    # ---------------------------------------------------------------- summary
    raw_means = np.array([r.data.mean(axis=1) for r in recs])           # DC offset per channel
    clean = {r.key: remove_dc(r.data, r.fs, highpass_hz=0.5) for r in recs}
    clean_std = np.array([clean[r.key].std(axis=1) for r in recs])
    print("=" * 64)
    print("STEW dataset summary")
    print("=" * 64)
    print(f"Subjects: {meta.subject_id.nunique()}   Recordings: {len(meta)} "
          f"({(meta.condition == 'lo').sum()} lo / {(meta.condition == 'hi').sum()} hi)")
    print(f"Array shape per recording: {recs[0].data.shape}   fs = {recs[0].fs} Hz   "
          f"duration = {recs[0].duration_s:.0f} s")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"Raw DC offset: mean {raw_means.mean():.0f} uV (range {raw_means.min():.0f}-{raw_means.max():.0f})")
    print(f"After DC removal + 0.5 Hz HP: channel std median {np.median(clean_std):.1f} uV "
          f"(5-95%: {np.percentile(clean_std, 5):.1f}-{np.percentile(clean_std, 95):.1f})")

    # ---------------------------------------------------------------- ratings
    rated = meta.dropna(subset=["rating"])
    missing = sorted(meta.loc[meta.rating.isna(), "subject_id"].unique().tolist())
    print(f"\nSubjects without a rating: {missing}  (kept for T1, masked for T2)")
    print("\nRating distribution (count of recordings):")
    dist = rated.pivot_table(index="rating", columns="condition", values="subject_id",
                             aggfunc="count", fill_value=0)[["lo", "hi"]]
    dist = dist.reindex(range(1, 10), fill_value=0)
    print(dist.to_string())

    print("\nT2 workload level by condition (rated recordings only):")
    levels = pd.crosstab(rated.condition, rated.label_level.map(dict(enumerate(LEVEL_NAMES))))
    levels = levels.reindex(index=["lo", "hi"], columns=LEVEL_NAMES, fill_value=0)
    print(levels.to_string())

    wide = rated.pivot(index="subject_id", columns="condition", values="rating")[["lo", "hi"]]
    diff = wide.hi - wide.lo
    stat = wilcoxon(wide.hi, wide.lo)
    lo_hi = {
        "n_subjects_rated": int(len(wide)),
        "rest_mean": float(wide.lo.mean()), "rest_std": float(wide.lo.std()), "rest_median": float(wide.lo.median()),
        "task_mean": float(wide.hi.mean()), "task_std": float(wide.hi.std()), "task_median": float(wide.hi.median()),
        "diff_mean": float(diff.mean()), "diff_std": float(diff.std()),
        "n_task_gt_rest": int((diff > 0).sum()), "n_equal": int((diff == 0).sum()), "n_task_lt_rest": int((diff < 0).sum()),
        "wilcoxon_stat": float(stat.statistic), "wilcoxon_p": float(stat.pvalue),
    }
    print("\nLo vs hi ratings (paired, rated subjects):")
    print(f"  rest: {lo_hi['rest_mean']:.2f} ± {lo_hi['rest_std']:.2f} (median {lo_hi['rest_median']:.0f})")
    print(f"  task: {lo_hi['task_mean']:.2f} ± {lo_hi['task_std']:.2f} (median {lo_hi['task_median']:.0f})")
    print(f"  task - rest: {lo_hi['diff_mean']:.2f} ± {lo_hi['diff_std']:.2f};  task>rest in "
          f"{lo_hi['n_task_gt_rest']}/{len(wide)}, equal {lo_hi['n_equal']}, lower {lo_hi['n_task_lt_rest']}")
    print(f"  Wilcoxon signed-rank: W={lo_hi['wilcoxon_stat']:.1f}, p={lo_hi['wilcoxon_p']:.2e}")
    print("  NOTE: T2 levels are strongly confounded with condition (rest is almost always 'low').")

    # ---------------------------------------------------------------- plots
    plot_ratings(dist)
    plot_raw(clean, subject=1)
    plot_raw(clean, subject=24)
    psd_freqs, psd_by_cond = grand_psd(recs, clean)
    plot_grand_psd(psd_freqs, psd_by_cond)
    plot_subject_psd(recs, clean, subjects=[1, 12, 24, 36])

    summary = {
        "n_subjects": int(meta.subject_id.nunique()), "n_recordings": int(len(meta)),
        "shape": list(recs[0].data.shape), "fs": recs[0].fs, "channels": CHANNELS,
        "raw_dc_offset_uv": {"mean": float(raw_means.mean()), "min": float(raw_means.min()), "max": float(raw_means.max())},
        "clean_channel_std_uv_median": float(np.median(clean_std)),
        "subjects_missing_rating": missing,
        "rating_distribution": {c: {int(k): int(v) for k, v in dist[c].items()} for c in ["lo", "hi"]},
        "level_by_condition": {c: {lv: int(levels.loc[c, lv]) for lv in LEVEL_NAMES} for c in ["lo", "hi"]},
        "lo_vs_hi_ratings": lo_hi,
    }
    with open(OUT / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    meta.to_csv(OUT / "recordings.csv", index=False)
    print(f"\nSaved plots + dataset_summary.json -> {OUT}")


def plot_ratings(dist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(1, 10)
    w = 0.38
    for i, c in enumerate(["lo", "hi"]):
        ax.bar(x + (i - 0.5) * (w + 0.02), dist[c].values, width=w, color=COLORS[c], label=NAMES[c])
    for edge in (3.5, 6.5):
        ax.axvline(edge, color=MUTED, lw=0.8, ls="--")
    for xc, name in zip((2, 5, 8), LEVEL_NAMES):
        ax.text(xc, ax.get_ylim()[1] * 0.95, name, ha="center", color=MUTED, fontsize=9)
    ax.set_xticks(x)
    ax.set_xlabel("Self-reported workload rating (1-9)")
    ax.set_ylabel("Recordings")
    ax.set_title("STEW workload ratings: rest vs task (45 rated subjects)", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.85))
    fig.tight_layout()
    fig.savefig(OUT / "rating_distribution.png", dpi=150)
    plt.close(fig)


def plot_raw(clean: dict, subject: int, seconds: float = 10.0, fs: int = 128) -> None:
    n = int(seconds * fs)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    both = [clean[f"sub{subject:02d}_{c}"][:, :n] for c in ("lo", "hi")]
    # Spacing from the typical channel std (robust to artifacts); clip extreme deflections
    # so one blink-laden channel cannot squash the others. Clipping is for display only.
    spacing = float(np.ceil(6 * np.median([b.std(axis=1) for b in both]) / 10) * 10)
    for ax, c, x in zip(axes, ["lo", "hi"], both):
        n_clipped = int((np.abs(x) > spacing).any(axis=1).sum())
        for k in range(x.shape[0]):
            ax.plot(t, np.clip(x[k], -spacing, spacing) - k * spacing, color=COLORS[c], lw=0.7)
        if n_clipped:
            ax.text(1, 1.01, f"{n_clipped} channel(s) clipped at ±{spacing:g} uV", ha="right",
                    transform=ax.transAxes, color=MUTED, fontsize=8)
        ax.set_yticks(-np.arange(len(CHANNELS)) * spacing, CHANNELS)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Subject {subject:02d} - {NAMES[c]}", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].text(0, 1.06, f"First {seconds:g} s, DC removed + 0.5 Hz high-pass; trace spacing {spacing:g} uV",
                 transform=axes[0].transAxes, color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"raw_sub{subject:02d}.png", dpi=150)
    plt.close(fig)


def grand_psd(recs, clean):
    """Channel-averaged Welch PSD (2-s segments) per recording, grouped by condition."""
    by_cond = {"lo": [], "hi": []}
    freqs = None
    for r in recs:
        freqs, p = welch(clean[r.key], fs=r.fs, nperseg=2 * r.fs, axis=1)
        by_cond[r.condition].append(10 * np.log10(p.mean(axis=0)))
    return freqs, {c: np.array(v) for c, v in by_cond.items()}


def plot_grand_psd(freqs, by_cond) -> None:
    keep = (freqs >= 0.5) & (freqs <= 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in ["lo", "hi"]:
        m, s = by_cond[c].mean(axis=0)[keep], by_cond[c].std(axis=0)[keep]
        ax.fill_between(freqs[keep], m - s, m + s, color=COLORS[c], alpha=0.15, lw=0)
        ax.plot(freqs[keep], m, color=COLORS[c], label=f"{NAMES[c]} (n={len(by_cond[c])})")
    for f0, name in [(4, "θ"), (8, "α"), (13, "β"), (30, "γ"), (45, "")]:
        ax.axvline(f0, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB re 1 uV²/Hz)")
    ax.set_title("Grand-average PSD, mean of 14 channels (band = ±1 SD across subjects)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "psd_grand_average.png", dpi=150)
    plt.close(fig)


def plot_subject_psd(recs, clean, subjects) -> None:
    fig, axes = plt.subplots(1, len(subjects), figsize=(3.2 * len(subjects), 3.2), sharey=True)
    for ax, s in zip(axes, subjects):
        for c in ["lo", "hi"]:
            f, p = welch(clean[f"sub{s:02d}_{c}"], fs=128, nperseg=256, axis=1)
            keep = (f >= 0.5) & (f <= 60)
            ax.plot(f[keep], 10 * np.log10(p.mean(axis=0)[keep]), color=COLORS[c], label=NAMES[c])
        ax.set_title(f"Subject {s:02d}", loc="left")
        ax.set_xlabel("Hz")
    axes[0].set_ylabel("Power (dB)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "psd_subjects.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
