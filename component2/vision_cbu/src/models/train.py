"""Data preparation and training for the baseline and temporal models.

Real-data entry points (`train_real`) refuse to run without valid real labelled features.
Demo entry points (`train_demo`) use synthetic data only and write to data/demo/models
with model names prefixed 'DEMO_ONLY_'.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.feature_schema import FEATURE_COLUMNS, SchemaError, validate_feature_frame
from src.data.sequence_builder import SequenceSet, build_sequences
from src.data.validate_dataset import RealDataUnavailableError, assert_no_split_leakage
from src.models.baseline import aggregate_windows, baseline_predict_proba, build_baseline
from src.models.labels import LabelScheme, get_label_scheme
from src.models.registry import save_model
from src.processing.preprocessing import Normalizer, forward_fill_within_video
from src.utils.config import resolve_path, set_seed
from src.utils.logging_config import get_logger

log = get_logger("train")


@dataclass
class PreparedData:
    train: SequenceSet
    val: SequenceSet
    test: SequenceSet
    normalizer: Normalizer
    scheme: LabelScheme
    feature_columns: list[str]
    data_source: str            # 'real' | 'synthetic'
    split_info: dict


# ---------------------------------------------------------------- data preparation
def assign_validation_split(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """If no validation split exists, carve one out of TRAIN at participant level."""
    if (df["split"] == "validation").any():
        return df
    train = df[df["split"] == "train"]
    groups = train["participant_id"].where(train["participant_id"] != "unknown", train["video_id"])
    if groups.nunique() < 2:
        raise RealDataUnavailableError("Not enough training participants to create a validation split.")
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    _, val_idx = next(gss.split(train, groups=groups))
    val_groups = set(groups.iloc[val_idx])
    df = df.copy()
    sel = (df["split"] == "train") & df["participant_id"].where(
        df["participant_id"] != "unknown", df["video_id"]).isin(val_groups)
    df.loc[sel, "split"] = "validation"
    return df


def data_identity(df: pd.DataFrame) -> dict:
    vids = sorted(df["video_id"].astype(str).unique())
    return {"n_videos": len(vids), "n_participants": int(df["participant_id"].nunique()),
            "video_id_sha256": hashlib.sha256("\n".join(vids).encode()).hexdigest()[:16],
            "split_videos": df.groupby("split")["video_id"].nunique().to_dict(),
            "split_participants": df.groupby("split")["participant_id"].nunique().to_dict()}


def build_split_sequences(cfg: dict, df: pd.DataFrame, scheme: LabelScheme,
                          feature_columns: list[str], window: int | None = None,
                          stride: int | None = None) -> dict[str, SequenceSet]:
    s = cfg["sequences"]
    window = window or s["window_size"]
    stride = stride or s["stride"]
    df = forward_fill_within_video(df, feature_columns, s.get("max_interpolate_gap", 3))
    out = {}
    for split in ("train", "validation", "test"):
        part = df[df["split"] == split]
        seq, stats = build_sequences(part, feature_columns, window, stride, s["min_valid_ratio"],
                                     label_column=scheme.column, label_transform=scheme.transform)
        log.info("%s: %d windows from %d videos (%d without windows)", split, len(seq),
                 stats["videos"], stats["videos_without_windows"])
        out[split] = seq
    return out


def apply_normalizer(seq: SequenceSet, norm: Normalizer) -> SequenceSet:
    X = norm.transform(seq.X) if len(seq) else seq.X.astype(np.float32)
    X = X * seq.mask[..., None]
    return SequenceSet(X, seq.mask, seq.y, seq.video_ids, seq.participant_ids, seq.splits,
                       seq.start_ts, seq.feature_columns, dict(seq.meta))


def prepare_data(cfg: dict, df: pd.DataFrame, data_source: str,
                 feature_columns: list[str] = FEATURE_COLUMNS) -> PreparedData:
    scheme = get_label_scheme(cfg)
    try:
        df = validate_feature_frame(df, allow_synthetic=(data_source == "synthetic"))
    except SchemaError as exc:
        raise RealDataUnavailableError(f"Feature table invalid: {exc}") from exc
    if data_source == "real" and (df["is_synthetic"].any() or (df["source"] == "synthetic").any()):
        raise RealDataUnavailableError("Synthetic rows found in real data - refusing to mix.")
    if data_source == "synthetic" and not df["is_synthetic"].all():
        raise ValueError("Demo training accepts synthetic rows only.")

    df = df[df[scheme.column].notna()].copy()
    if df.empty:
        raise RealDataUnavailableError(f"No rows with labels in {scheme.column}.")
    df = assign_validation_split(df, cfg["split"]["validation_fraction"], cfg["project"]["seed"])
    missing_splits = {"train", "validation", "test"} - set(df["split"].unique())
    if missing_splits:
        raise RealDataUnavailableError(f"Missing splits: {sorted(missing_splits)}")

    assert_no_split_leakage(df, group_cols=("video_id",))
    warnings = []
    try:
        assert_no_split_leakage(df, group_cols=("participant_id",))
    except ValueError as exc:
        if cfg["split"].get("participant_level", True) and not cfg["split"].get("use_official_splits", True):
            raise
        warnings.append(f"{exc} (official split preserved; reported as limitation)")
        log.warning(warnings[-1])

    seqs = build_split_sequences(cfg, df, scheme, feature_columns)
    if len(seqs["train"]) == 0 or len(seqs["test"]) == 0:
        raise RealDataUnavailableError("No usable windows in train or test split (check quality / window size).")
    if len(seqs["validation"]) == 0:
        raise RealDataUnavailableError("No usable windows in validation split.")
    tr = seqs["train"]
    norm = Normalizer(cfg["preprocessing"]["normalization"]).fit(tr.X[tr.mask > 0.5])   # TRAIN ONLY
    split_info = {**data_identity(df), "warnings": warnings,
                  "windows": {k: len(v) for k, v in seqs.items()},
                  "class_counts": {k: np.bincount(v.y, minlength=len(scheme.class_names)).tolist()
                                   for k, v in seqs.items()}}
    return PreparedData(apply_normalizer(seqs["train"], norm), apply_normalizer(seqs["validation"], norm),
                        apply_normalizer(seqs["test"], norm), norm, scheme, list(feature_columns),
                        data_source, split_info)


# ---------------------------------------------------------------- shared metadata
def _base_meta(cfg: dict, data: PreparedData) -> dict:
    return {"data_source": data.data_source, "demo_only": data.data_source != "real",
            "dataset": "DAiSEE" if data.data_source == "real" else "SYNTHETIC (demo only)",
            "feature_columns": data.feature_columns, "label_scheme": data.scheme.to_dict(),
            "label_policy": cfg["labels"].get("video_label_policy"),
            "window": cfg["sequences"]["window_size"], "stride": cfg["sequences"]["stride"],
            "sampling_fps": cfg["extraction"]["target_fps"],
            "min_valid_ratio": cfg["sequences"]["min_valid_ratio"],
            "normalizer": data.normalizer.to_dict(), "training_data": data.split_info,
            "seed": cfg["project"]["seed"]}


def _name(prefix: str, kind: str, data_source: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "DEMO_ONLY_" if data_source != "real" else ""
    return f"{tag}{prefix}_{kind}_{stamp}"


def class_weights(y: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n).astype(np.float64)
    w = np.where(counts > 0, len(y) / (n * np.maximum(counts, 1)), 0.0)
    return w


# ---------------------------------------------------------------- baseline
def train_baseline(cfg: dict, data: PreparedData, out_root: Path, feature_subset: list[str] | None = None,
                   save: bool = True) -> tuple[object, dict]:
    set_seed(cfg["project"]["seed"])
    tr, va = data.train, data.val
    cols_idx = None
    if feature_subset is not None:
        cols_idx = [data.feature_columns.index(c) for c in feature_subset]
    Xtr = tr.X if cols_idx is None else tr.X[:, :, cols_idx]
    model = build_baseline(cfg, cfg["project"]["seed"])
    t0 = time.perf_counter()
    model.fit(aggregate_windows(Xtr, tr.mask), tr.y)
    secs = time.perf_counter() - t0
    Xva = va.X if cols_idx is None else va.X[:, :, cols_idx]
    val_acc = float((baseline_predict_proba(model, Xva, va.mask, len(data.scheme.class_names)).argmax(1) == va.y).mean())
    meta = _base_meta(cfg, data)
    meta.update(model_params={"type": cfg["baseline"]["type"], "class_weight": cfg["baseline"]["class_weight"],
                              "aggregation": ["mean", "std", "min", "max"]},
                history={"train_seconds": round(secs, 3), "val_accuracy": val_acc})
    if save:
        name = _name(cfg["baseline"]["type"], "baseline", data.data_source)
        path = save_model(out_root, name, "baseline", model, meta)
        meta["path"] = str(path)
        meta["model_name"] = name
        log.info("Saved baseline to %s (validation accuracy %.3f)", path, val_acc)
    return model, meta


# ---------------------------------------------------------------- temporal
def train_temporal(cfg: dict, data: PreparedData, out_root: Path, save: bool = True) -> tuple[object, dict]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.temporal_model import TemporalClassifier, pick_device

    tc = cfg["temporal"]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = pick_device(tc.get("device", "auto"))
    n_classes = len(data.scheme.class_names)
    params = dict(n_features=len(data.feature_columns), n_classes=n_classes, hidden_size=int(tc["hidden_size"]),
                  num_layers=int(tc["num_layers"]), cell=tc["cell"], dropout=float(tc["dropout"]),
                  bidirectional=bool(tc["bidirectional"]))
    model = TemporalClassifier(**params).to(device)

    def loader(s: SequenceSet, shuffle: bool):
        ds = TensorDataset(torch.from_numpy(s.X.astype(np.float32)), torch.from_numpy(s.mask.astype(np.float32)),
                           torch.from_numpy(s.y.astype(np.int64)))
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=int(tc["batch_size"]), shuffle=shuffle, generator=g)

    tl, vl = loader(data.train, True), loader(data.val, False)
    w = torch.tensor(class_weights(data.train.y, n_classes), dtype=torch.float32, device=device) \
        if tc.get("class_weight") == "balanced" else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]), weight_decay=float(tc["weight_decay"]))

    def run(dl, train: bool):
        model.train(train)
        tot, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for xb, mb, yb in dl:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                logits = model(xb, mb)
                loss = loss_fn(logits, yb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                tot += loss.item() * len(yb)
                correct += int((logits.argmax(1) == yb).sum())
                n += len(yb)
        return tot / max(n, 1), correct / max(n, 1)

    history, best, best_state, bad = [], float("inf"), None, 0
    t0 = time.perf_counter()
    for epoch in range(1, int(tc["epochs"]) + 1):
        tr_loss, tr_acc = run(tl, True)
        va_loss, va_acc = run(vl, False)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        log.info("epoch %3d  train_loss %.4f  acc %.3f | val_loss %.4f  acc %.3f", epoch, tr_loss, tr_acc, va_loss, va_acc)
        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tc["patience"]):
                log.info("Early stopping at epoch %d", epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    meta = _base_meta(cfg, data)
    meta.update(model_params=params,
                training_params={k: tc[k] for k in ("batch_size", "learning_rate", "weight_decay", "epochs",
                                                    "patience", "class_weight")},
                history={"epochs": history, "best_val_loss": best, "train_seconds": round(time.perf_counter() - t0, 2),
                         "device": str(device)})
    if save:
        name = _name(tc["cell"], "temporal", data.data_source)
        path = save_model(out_root, name, "temporal", model, meta)
        meta["path"] = str(path)
        meta["model_name"] = name
        log.info("Saved temporal model to %s", path)
    return model, meta


# ---------------------------------------------------------------- entry points
def load_real_features(cfg: dict) -> pd.DataFrame:
    from src.data.extract_features import load_feature_table
    df = load_feature_table(cfg)
    if df.empty:
        raise RealDataUnavailableError(
            "No real extracted features in data/processed/features/. Add DAiSEE to data/raw/daisee/, "
            "run scripts/audit_dataset.py, then scripts/extract_features.py.")
    return df


def load_demo_features(cfg: dict) -> pd.DataFrame:
    p = resolve_path(cfg, "demo_dir") / "synthetic_features.csv"
    if not p.exists():
        raise FileNotFoundError("No synthetic demo data. Run: python scripts/generate_demo_data.py")
    return pd.read_csv(p, dtype={"participant_id": str, "video_id": str})


def train_real(cfg: dict, kind: str) -> dict:
    data = prepare_data(cfg, load_real_features(cfg), "real")
    root = resolve_path(cfg, "models_dir")
    fn = train_baseline if kind == "baseline" else train_temporal
    return fn(cfg, data, root)[1]


def train_demo(cfg: dict, kind: str) -> dict:
    data = prepare_data(cfg, load_demo_features(cfg), "synthetic")
    root = resolve_path(cfg, "demo_dir") / "models"
    fn = train_baseline if kind == "baseline" else train_temporal
    return fn(cfg, data, root)[1]
