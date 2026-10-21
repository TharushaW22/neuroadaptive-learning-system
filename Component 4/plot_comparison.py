"""Plot the 4-algorithm comparison from results/*.csv."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                # headless-friendly
import matplotlib.pyplot as plt

RESULTS = "results"
OUT = os.path.join(RESULTS, "plots")
RAW = os.path.join(RESULTS, "comparison_raw.csv")
SUM = os.path.join(RESULTS, "comparison_summary.csv")


def main():
    if not os.path.exists(RAW) or not os.path.exists(SUM):
        raise FileNotFoundError("Run comparison.py first (needs comparison_raw.csv & comparison_summary.csv).")
    os.makedirs(OUT, exist_ok=True)

    raw = pd.read_csv(RAW)
    agg = pd.read_csv(SUM).sort_values("score", ascending=False).reset_index(drop=True)

    palette = {
        "linucb":    "#4F46E5",
        "qlearning": "#059669",
        "bkt":       "#DC2626",
        "topsis":    "#D97706",
    }
    colors = [palette.get(a, "#64748B") for a in agg["algorithm"]]

    # ------------------------------------------------------------
    # Fig 1 — three-panel bar chart with error bars
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy (vs expert)", (0, 1)),
        ("gain_mean",     "gain_std",     "Mean Learning Gain",   (0, 1)),
        ("lat_mean",      "lat_std",      "Latency (ms)",         None),
    ]
    for ax, (m, s, title, ylim) in zip(axes, metrics):
        ax.bar(agg["algorithm"], agg[m], yerr=agg[s],
               color=colors, edgecolor="black", linewidth=0.6,
               capsize=4, alpha=0.9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(m.replace("_mean", ""))
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("MD-AP2L — 4-Algorithm Comparison (mean ± std over 10 seeds)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1_bars.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------
    # Fig 2 — per-seed boxplot (spread across seeds, not just mean)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [
        (axes[0], "accuracy",   "Accuracy per seed"),
        (axes[1], "learning_gain", "Learning gain per seed"),
    ]:
        data = [raw.loc[raw["algorithm"] == a, col].values for a in agg["algorithm"]]
        bp = ax.boxplot(data, labels=agg["algorithm"], patch_artist=True,
                        medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    fig.suptitle("Per-seed distribution (stability)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2_boxplot.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------
    # Fig 3 — radar chart of normalised metrics
    # ------------------------------------------------------------
    metrics_norm = agg.copy()
    for c in ["accuracy_mean", "gain_mean"]:
        metrics_norm[c] = metrics_norm[c] / metrics_norm[c].max()
    metrics_norm["lat_score"] = 1.0 - metrics_norm["lat_mean"] / 50.0
    metrics_norm["lat_score"] = metrics_norm["lat_score"].clip(lower=0.0)
    metrics_norm["score_norm"] = metrics_norm["score"] / metrics_norm["score"].max()

    dims = ["accuracy_mean", "gain_mean", "lat_score", "score_norm"]
    labels = ["Accuracy", "Learning Gain", "Speed (1 - lat/50ms)", "Weighted Score"]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for _, row in metrics_norm.iterrows():
        vals = [row[d] for d in dims] + [row[dims[0]]]
        ax.plot(angles, vals, linewidth=2,
                color=palette.get(row["algorithm"], "#64748B"),
                label=row["algorithm"])
        ax.fill(angles, vals, alpha=0.10,
                color=palette.get(row["algorithm"], "#64748B"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalised comparison (higher = better)",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "3_radar.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Fig 4 — per-seed accuracy heatmap (algorithm × seed)
    # ------------------------------------------------------------
    pivot = raw.pivot(index="algorithm", columns="seed", values="accuracy")
    pivot = pivot.reindex(agg["algorithm"])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("seed")
    ax.set_title("Accuracy by seed (the winner should win on most columns)",
                 fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if pivot.values[i, j] < 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "4_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"[OK] Wrote 4 plots to {OUT}/")
    print(f"     winner: {agg.loc[0, 'algorithm']}  (score={agg.loc[0, 'score']:.4f})")


if __name__ == "__main__":
    main()