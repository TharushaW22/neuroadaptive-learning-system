"""Matplotlib figures for evaluation reports (headless, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _banner(fig, text: str | None):
    if text:
        fig.text(0.5, 0.995, text, ha="center", va="top", fontsize=9, color="#b00020", weight="bold")


def plot_confusion_matrix(cm, class_names, title: str, path: Path, banner: str | None = None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.0 * len(class_names) + 2.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thr = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _banner(fig, banner)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_training_curves(history: list[dict], title: str, path: Path, banner: str | None = None):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="validation")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, [h["train_acc"] for h in history], label="train")
    a2.plot(ep, [h["val_acc"] for h in history], label="validation")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.suptitle(title)
    _banner(fig, banner)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(rows: list[dict], path: Path, banner: str | None = None,
                          metrics=("accuracy", "macro_f1")):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(rows))
    w = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, [r[m] for r in rows], w, label=m)
    ax.set_xticks(x + w * (len(metrics) - 1) / 2, [r["model"] for r in rows])
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Model comparison (held-out test)")
    _banner(fig, banner)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
