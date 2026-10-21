"""Write experiment outputs and build summary tables FROM the saved result files.

results/<experiment_id>/
    metrics.json         full result dict (config + all metric views)
    folds.csv            per-fold metrics
    predictions.csv      out-of-fold predictions (one row per epoch or sequence)
    confusion_matrix.png pooled + recording-level confusion matrices
    config.yaml          the exact configuration of the run
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from eegengine.config import RESULTS_DIR


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save_experiment(out_dir: Path, result: dict, folds: pd.DataFrame, preds: pd.DataFrame | None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _clean(result)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    folds.to_csv(out_dir / "folds.csv", index=False)
    if preds is not None:
        preds.to_csv(out_dir / "predictions.csv", index=False, float_format="%.5f")
    cfg_keys = ("experiment_id", "kind", "task", "model", "feature_set", "families", "normalisation", "cv",
                "seed", "selection", "exclude_bad_windows", "model_config", "train_config")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: clean[k] for k in cfg_keys if k in clean}, f, sort_keys=False)
    plot_confusions(out_dir / "confusion_matrix.png", result)


def plot_confusions(path: Path, result: dict) -> None:
    classes = result["classes"]
    mats = [("pooled", result.get("confusion_pooled")), ("per recording", result.get("confusion_recording"))]
    mats = [(t, np.array(m)) for t, m in mats if m is not None]
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (title, M) in zip(axes, mats):
        norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "#0b0b0b")
        ax.set_xticks(range(len(classes)), classes, fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle(result["experiment_id"], fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_results(kind: str | None = None, root: Path = RESULTS_DIR) -> list[dict]:
    """All metrics.json files under results/ (optionally of one kind)."""
    out = []
    for p in sorted(root.glob("*/metrics.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if kind is None or r.get("kind") == kind:
            out.append(r)
    return out


def fmt(ms: dict | None, key: str = "macro_f1") -> str:
    """'0.812 ± 0.104' from a per_fold dict."""
    if not ms or key not in ms or ms[key]["mean"] is None:
        return "—"
    return f"{ms[key]['mean']:.3f} ± {ms[key]['std']:.3f}"


def num(v) -> str:
    return "—" if v is None else f"{v:.3f}"
