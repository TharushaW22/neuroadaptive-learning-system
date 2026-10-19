"""Deep model training for one CV fold (and for the final deployable model).

Per fold:
  1. split the training subjects into fit / validation subjects (by subject, `inner_validation_split`)
  2. fit the input scalers on the FIT subjects only (per-channel amplitude scale; node-feature z-score)
  3. train with AdamW + class-weighted masked multi-task loss + augmentation; every epoch draws a fresh
     random set of sequence start points per recording (same cost per epoch, more variety)
  4. early stopping on validation loss; restore the best weights
  5. temperature scaling fit on the validation logits
  6. predict the test sequences (raw and calibrated probabilities)

Epoch-level models (EEGNet) train on single epochs; their test predictions are averaged over the
same test sequences as the sequence models, so every model is scored on identical 18-s windows.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from eegengine.models.deep.eegnet import EEGNet
from eegengine.models.deep.hybrid import HybridModel, masked_multitask_loss
from eegengine.training.calibration import fit_temperature, softmax
from eegengine.training.cv import assert_no_subject_overlap, inner_validation_split
from eegengine.training.sequences import recording_groups, sample_items, strided_items  # noqa: F401


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
@dataclass
class Scalers:
    """Statistics learned from the training (fit) subjects only."""

    channel_scale: np.ndarray       # (C,) robust amplitude scale per channel
    node_mean: np.ndarray           # (C, 5) mean of log band power
    node_std: np.ndarray            # (C, 5)
    clip: float = 20.0

    @classmethod
    def fit(cls, X: np.ndarray, logbp: np.ndarray, rows: np.ndarray, clip: float = 20.0) -> "Scalers":
        # median over epochs of each epoch's per-channel std: robust to a few artifact epochs
        scale = np.median(X[rows].std(axis=-1), axis=0) + 1e-6
        return cls(scale.astype(np.float32), logbp[rows].mean(0).astype(np.float32),
                   (logbp[rows].std(0) + 1e-6).astype(np.float32), clip)

    def to_dict(self) -> dict:
        return {"channel_scale": self.channel_scale.tolist(), "node_mean": self.node_mean.tolist(),
                "node_std": self.node_std.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "Scalers":
        return cls(np.array(d["channel_scale"], np.float32), np.array(d["node_mean"], np.float32),
                   np.array(d["node_std"], np.float32), d["clip"])


class DeepInputs:
    """Holds all epochs + graph inputs; builds scaled torch batches from index arrays (B, L)."""

    def __init__(self, X: np.ndarray, bandpower: np.ndarray, adj: np.ndarray):
        self.X = X                                         # (N, C, T) float32, uV
        self.logbp = np.log10(bandpower + 1e-12).astype(np.float32)   # (N, C, 5)
        self.adj = adj.astype(np.float32)                  # (N, C, C)
        self.scalers: Scalers | None = None

    def batch(self, idx: np.ndarray, augment: dict | None = None, rng: np.random.Generator | None = None):
        s = self.scalers
        x = self.X[idx] / s.channel_scale[None, None, :, None]          # (B, L, C, T)
        x = np.clip(x, -s.clip, s.clip)
        node = (self.logbp[idx] - s.node_mean) / s.node_std
        adj = self.adj[idx]
        if augment:
            x = augment_batch(x, augment, rng)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.from_numpy(node.astype(np.float32)), torch.from_numpy(adj))


def augment_batch(x: np.ndarray, aug: dict, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation on scaled inputs: Gaussian noise, channel dropout, small time shift."""
    x = x.copy()
    B, L, C, T = x.shape
    if aug.get("noise_std", 0) > 0:
        x += rng.normal(0, aug["noise_std"], size=x.shape).astype(np.float32)
    p = aug.get("channel_dropout", 0)
    if p > 0:
        drop = rng.random((B, 1, C, 1)) < p                  # same channel dropped in all epochs
        x = np.where(drop, 0.0, x)
    k = aug.get("max_shift", 0)
    if k > 0:
        for b in range(B):
            x[b] = np.roll(x[b], int(rng.integers(-k, k + 1)), axis=-1)
    return x


def edge_matrix(arrays: dict, source: str = "plv", band: str | int = "mean") -> np.ndarray:
    """Edge weights for the GNN: (N, 4, C, C) PLV/coherence -> (N, C, C)."""
    M = arrays[source]
    return M.mean(axis=1) if band == "mean" else M[:, int(band)]


# ---------------------------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------------------------
def build_model(spec: dict, model_cfg: dict, n_classes: int, n_samples: int) -> torch.nn.Module:
    if spec["type"] == "eegnet":
        c = model_cfg.get("cnn", {})
        return EEGNet(n_classes, n_samples=n_samples, F1=c.get("F1", 8), D=c.get("D", 2), F2=c.get("F2", 16),
                      kernel=c.get("kernel", 32), dropout=c.get("dropout", 0.25))
    if spec["type"] == "hybrid":
        mc = {k: v for k, v in model_cfg.items()}
        return HybridModel(n_classes, branches=tuple(spec["branches"]), n_samples=n_samples, **mc)
    raise ValueError(spec["type"])


def is_epoch_model(spec: dict) -> bool:
    return spec["type"] == "eegnet"


def forward(model, spec, x, node, adj):
    if is_epoch_model(spec):
        return model(x[:, 0])                       # (B, 1, C, T) -> single epoch
    return model(x, node, adj)


# ---------------------------------------------------------------------------------------------
# Training items
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, spec, inputs: DeepInputs, items: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(items), batch_size):
        x, node, adj = inputs.batch(items[i:i + batch_size])
        outs.append(forward(model, spec, x, node, adj)["workload"].numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1))


def train_model(spec: dict, model_cfg: dict, train_cfg: dict, aug_cfg: dict, inputs: DeepInputs, meta,
                y_epoch: np.ndarray, fit_rows: np.ndarray, val_rows: np.ndarray, n_classes: int,
                seed: int, log=print):
    """Train one model. Returns (model, temperature, history, val_items, val_logits)."""
    set_torch_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(train_cfg.get("threads", 8))
    L = 1 if is_epoch_model(spec) else train_cfg["seq_len"]

    inputs.scalers = Scalers.fit(inputs.X, inputs.logbp, fit_rows, train_cfg.get("clip", 20.0))
    fit_groups = recording_groups(meta, fit_rows)
    val_items = strided_items(recording_groups(meta, val_rows), L, max(1, train_cfg["val_stride"] if L > 1 else 2))
    val_y = y_epoch[val_items[:, -1]]

    # class weights from the fit recordings (inverse frequency, normalised to mean 1)
    rec_y = np.array([y_epoch[g[0]] for g in fit_groups])
    counts = np.bincount(rec_y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    class_w = torch.tensor(w, dtype=torch.float32)

    model = build_model(spec, model_cfg, n_classes, inputs.X.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    n_per_rec = train_cfg["epoch_items_per_recording"] * (train_cfg.get("epoch_model_multiplier", 4) if L == 1 else 1)
    best = {"loss": np.inf, "state": None, "epoch": -1}
    history = []
    patience = train_cfg["patience"]
    bs = train_cfg["batch_size"] * (train_cfg.get("epoch_model_multiplier", 4) if L == 1 else 1)

    for ep in range(train_cfg["max_epochs"]):
        t0 = time.perf_counter()
        model.train()
        items = sample_items(fit_groups, L, n_per_rec, rng)
        rng.shuffle(items)
        tot, nb = 0.0, 0
        for i in range(0, len(items), bs):
            it = items[i:i + bs]
            x, node, adj = inputs.batch(it, aug_cfg, rng)
            yb = torch.from_numpy(y_epoch[it[:, -1]].astype(np.int64))
            out = forward(model, spec, x, node, adj)
            masks = {"workload": torch.ones(len(yb), dtype=torch.bool)}   # aux heads: no labels on STEW
            loss = masked_multitask_loss(out, {"workload": yb}, masks, {"workload": class_w})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            opt.step()
            tot += loss.item()
            nb += 1
        vl = predict_logits(model, spec, inputs, val_items)
        val_loss = float(F.cross_entropy(torch.from_numpy(vl), torch.from_numpy(val_y.astype(np.int64))))
        val_acc = float((vl.argmax(1) == val_y).mean())
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": val_loss, "val_acc": val_acc,
                        "seconds": round(time.perf_counter() - t0, 1)})
        log(f"    ep {ep:02d} train {tot / max(nb, 1):.4f} val {val_loss:.4f} acc {val_acc:.3f} "
            f"[{time.perf_counter() - t0:.0f}s]")
        if val_loss < best["loss"] - 1e-4:
            best = {"loss": val_loss, "state": copy.deepcopy(model.state_dict()), "epoch": ep}
        elif ep - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    val_logits = predict_logits(model, spec, inputs, val_items)
    # Temperature scaling needs both classes in the validation set; otherwise it degenerates
    # (it can push T -> 0 and make the model over-confident), so keep T = 1.
    T = fit_temperature(val_logits, val_y) if len(np.unique(val_y)) > 1 else 1.0
    return model, T, history, best["epoch"], val_items, val_logits, val_y


def run_fold(spec, model_cfg, train_cfg, aug_cfg, inputs, meta, y_epoch, groups_epoch, train_rows, test_rows,
             test_items, n_classes, seed, log=print) -> dict:
    """Train on train_rows (split into fit/val by subject) and predict the given test sequences."""
    # subject-wise validation split; re-draw (deterministically) until it contains every class, so
    # early stopping and temperature scaling see both classes (matters for T3: one class per subject)
    n_cls = len(np.unique(y_epoch[train_rows]))
    for k in range(50):
        fit_local, val_local = inner_validation_split(groups_epoch[train_rows], train_cfg.get("val_frac", 0.15), seed + 1000 * k)
        if len(np.unique(y_epoch[train_rows][val_local])) == n_cls:
            break
    fit_rows, val_rows = train_rows[fit_local], train_rows[val_local]
    assert_no_subject_overlap(groups_epoch[fit_rows], groups_epoch[test_rows])
    assert_no_subject_overlap(groups_epoch[val_rows], groups_epoch[test_rows])
    t0 = time.perf_counter()
    model, T, history, best_ep, val_items, val_logits, val_y = train_model(
        spec, model_cfg, train_cfg, aug_cfg, inputs, meta, y_epoch, fit_rows, val_rows, n_classes, seed, log)
    train_s = time.perf_counter() - t0

    if is_epoch_model(spec):
        # predict every test epoch once, then average calibrated probabilities over each sequence
        uniq = np.unique(test_items)
        lg = predict_logits(model, spec, inputs, uniq[:, None])
        pos = {e: i for i, e in enumerate(uniq)}
        map_idx = np.vectorize(pos.get)(test_items)
        p_raw = softmax(lg)[map_idx].mean(axis=1)
        p_cal = softmax(lg, T)[map_idx].mean(axis=1)
        test_logits = None
    else:
        test_logits = predict_logits(model, spec, inputs, test_items)
        p_raw, p_cal = softmax(test_logits), softmax(test_logits, T)
    return {"model": model, "temperature": T, "history": history, "best_epoch": best_ep, "train_s": train_s,
            "p_raw": p_raw, "p_cal": p_cal, "test_logits": test_logits, "val_logits": val_logits, "val_y": val_y,
            "scalers": inputs.scalers, "n_fit_subjects": int(len(np.unique(groups_epoch[fit_rows]))),
            "n_val_subjects": int(len(np.unique(groups_epoch[val_rows])))}


def measure_latency(model, spec, inputs: DeepInputs, seq_len: int, n: int = 30, threads: int = 1) -> dict:
    """CPU latency of one window: one epoch (EEGNet) or one sequence of `seq_len` epochs (others)."""
    torch.set_num_threads(threads)
    model.eval()
    L = 1 if is_epoch_model(spec) else seq_len
    item = np.arange(L)[None, :]
    x, node, adj = inputs.batch(item)
    times = []
    with torch.no_grad():
        for _ in range(5):
            forward(model, spec, x, node, adj)
        for _ in range(n):
            t = time.perf_counter()
            forward(model, spec, x, node, adj)
            times.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "threads": threads,
            "window_epochs": L}
