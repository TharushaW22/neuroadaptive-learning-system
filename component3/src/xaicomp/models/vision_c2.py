"""Component 2's vision model, normaliser and window builder — minimal COPIES, read-only use.

Component 2 (component2/vision_cbu/) is read-only for Component 3 and its top-level package is called
`src`, which would clash on sys.path. Importing it would also compile bytecode next to its files.
The few definitions Component 3 needs are therefore copied here, unchanged in behaviour, with the
source cited. tests/test_step15_vision_c2_copies.py loads the ORIGINAL files and checks that these
copies give identical outputs.

  TemporalClassifier      component2/vision_cbu/src/models/temporal_model.py:8-41   (commit 6c7607e)
  Normalizer              component2/vision_cbu/src/processing/preprocessing.py:29-78 (transform/from_dict only;
                          fitting is never needed: C3 always uses C2's stored training-split parameters)
  forward_fill_within_video  same file, lines 16-26
  windows_for_video       component2/vision_cbu/src/data/sequence_builder.py:66-90
  clean_timestamps        same file, lines 57-63
  build_windows           adapted from build_sequences (same file, lines 93-131): identical windows, plus the
                          share of quality-OK frames per window (C3 uses it as the "noisy frames" covariate)

Nothing here writes files. Model checkpoints are read with torch.load(weights_only=True) into a new object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# component2/vision_cbu/src/data/feature_schema.py:40-47 (schema 1.0.0)
C2_FEATURE_COLUMNS = ["ear_left", "ear_right", "ear_mean", "blink_event", "blink_rate_per_min",
                      "gaze_h", "gaze_v", "head_yaw", "head_pitch", "head_roll",
                      "mouth_aspect_ratio", "brow_raise", "face_scale", "face_center_x", "face_center_y"]
C2_SCHEMA_VERSION = "1.0.0"


# ----------------------------------------------------------------------------- model (copy)
class TemporalClassifier(nn.Module):
    """Input (B,T,F) + mask (B,T) -> logits (B,C).  COPY of C2 temporal_model.py:8-41.

    The representation concatenates the masked mean of RNN outputs and the output at the
    last valid timestep, so padded steps never influence the prediction.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64, num_layers: int = 1,
                 cell: str = "gru", dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.config = dict(n_features=n_features, n_classes=n_classes, hidden_size=hidden_size,
                           num_layers=num_layers, cell=cell, dropout=dropout, bidirectional=bidirectional)
        self.rnn = rnn_cls(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * out_dim, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B,T,F), got {tuple(x.shape)}")
        if x.shape[-1] != self.config["n_features"]:
            raise ValueError(f"Expected {self.config['n_features']} features, got {x.shape[-1]}")
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device)
        x = torch.nan_to_num(x, nan=0.0) * mask.unsqueeze(-1)
        out, _ = self.rnn(x)                               # (B,T,H)
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        last_idx = (mask.sum(1).long() - 1).clamp(min=0)
        last = out[torch.arange(out.shape[0], device=x.device), last_idx]
        return self.head(torch.cat([mean, last], dim=1))


# ----------------------------------------------------------------------------- normaliser (copy)
class Normalizer:
    """Apply C2's stored normaliser. COPY of preprocessing.py:29-78 (transform + from_dict)."""

    def __init__(self, method: str = "standard"):
        if method not in ("standard", "minmax", "none"):
            raise ValueError(f"Unknown normalisation: {method}")
        self.method = method
        self.fill_values = self.a = self.b = None
        self.fitted = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN with training medians, then scale. Output contains no NaN."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit on training data first")
        X = X.astype(np.float64).copy()
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.broadcast_to(self.fill_values, X.shape)[nan]
        return ((X - self.a) / self.b).astype(np.float32)

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        n = cls(d["method"])
        n.fill_values, n.a, n.b = (np.array(d[k], dtype=np.float64) for k in ("fill_values", "a", "b"))
        n.fitted = True
        return n


# ----------------------------------------------------------------------------- windows (copies)
def forward_fill_within_video(df: pd.DataFrame, feature_cols: list[str], max_gap: int,
                              group_col: str = "video_id") -> pd.DataFrame:
    """Forward-fill gaps of at most max_gap samples, never across videos. COPY of preprocessing.py:16-26."""
    out = df.sort_values([group_col, "timestamp_s"]).copy()
    if "quality_ok" in out.columns:
        bad = ~out["quality_ok"].astype(bool)
        out.loc[bad, feature_cols] = np.nan
    if max_gap and max_gap > 0:
        out[feature_cols] = out.groupby(group_col, sort=False)[feature_cols].ffill(limit=max_gap)
    return out


def clean_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN/inf/negative or duplicate timestamps; sort. COPY of sequence_builder.py:57-63."""
    ts = pd.to_numeric(df["timestamp_s"], errors="coerce")
    ok = ts.notna() & np.isfinite(ts) & (ts >= 0)
    out = df.loc[ok].assign(timestamp_s=ts[ok]).sort_values("timestamp_s")
    out = out.drop_duplicates("timestamp_s", keep="first")
    return out, int(len(df) - len(out))


def windows_for_video(values: np.ndarray, quality_ok: np.ndarray, window: int, stride: int,
                      min_valid_ratio: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """[(start, X(T,F), mask(T,))] for one video. COPY of sequence_builder.py:66-90."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    n, f = values.shape
    out = []
    if n == 0:
        return out
    starts = list(range(0, max(n - window, 0) + 1, stride)) if n >= window else [0]
    for s in starts:
        seg = values[s:s + window]
        q = quality_ok[s:s + window].astype(bool)
        X = np.zeros((window, f), dtype=np.float32)
        m = np.zeros(window, dtype=np.float32)
        X[:len(seg)] = seg
        m[:len(seg)] = 1.0
        if q.sum() / window < min_valid_ratio:
            continue
        out.append((s, X, m))
    return out


@dataclass
class WindowSet:
    """Windows of one or more splits (raw, before normalisation) and their identifiers."""
    X: np.ndarray                 # (N, T, F) float32, may contain NaN
    mask: np.ndarray              # (N, T) float32
    y_level: np.ndarray           # (N,) original DAiSEE level 0-3
    video_ids: np.ndarray
    participant_ids: np.ndarray
    splits: np.ndarray
    start_frame: np.ndarray
    start_ts: np.ndarray
    quality_share: np.ndarray     # (N,) share of quality-OK frames among the T frames
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])


def build_windows(df: pd.DataFrame, feature_columns: list[str], window: int, stride: int, min_valid_ratio: float,
                  label_column: str) -> WindowSet:
    """Windows for every video in df, exactly as C2's build_sequences (sequence_builder.py:93-131):
    one label per video (videos with 0 or >1 distinct labels are skipped), windows never cross videos.
    `df` must already be forward-filled (C2 build_split_sequences, train.py:77)."""
    rows: dict[str, list] = {k: [] for k in ("X", "m", "y", "vid", "pid", "split", "start", "ts", "q")}
    for vid, g in df.groupby("video_id", sort=True):
        g, _ = clean_timestamps(g)
        labs = g[label_column].dropna().unique()
        if len(labs) != 1:
            continue
        values = g[feature_columns].to_numpy(dtype=np.float32)
        qok = g["quality_ok"].to_numpy()
        ts = g["timestamp_s"].to_numpy()
        for s, X, m in windows_for_video(values, qok, window, stride, min_valid_ratio):
            rows["X"].append(X); rows["m"].append(m); rows["y"].append(int(labs[0]))
            rows["vid"].append(str(vid)); rows["pid"].append(str(g["participant_id"].iloc[0]))
            rows["split"].append(str(g["split"].iloc[0])); rows["start"].append(int(s)); rows["ts"].append(float(ts[s]))
            rows["q"].append(float(qok[s:s + window].astype(bool).sum() / window))
    F = len(feature_columns)
    return WindowSet(
        X=np.stack(rows["X"]) if rows["X"] else np.zeros((0, window, F), np.float32),
        mask=np.stack(rows["m"]) if rows["m"] else np.zeros((0, window), np.float32),
        y_level=np.array(rows["y"], dtype=np.int64), video_ids=np.array(rows["vid"], dtype=str),
        participant_ids=np.array(rows["pid"], dtype=str), splits=np.array(rows["split"], dtype=str),
        start_frame=np.array(rows["start"], dtype=np.int64), start_ts=np.array(rows["ts"], dtype=np.float64),
        quality_share=np.array(rows["q"], dtype=np.float64),
        meta={"window": window, "stride": stride, "min_valid_ratio": min_valid_ratio, "label_column": label_column})


# ----------------------------------------------------------------------------- checkpoints
class VisionModelError(RuntimeError):
    """The C2 checkpoint is missing or not the model Component 3 expects."""


def read_metadata(model_dir: Path | str) -> dict:
    p = Path(model_dir) / "metadata.json"
    if not p.exists():
        raise VisionModelError(f"no metadata.json in {model_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def metadata_problems(meta: dict, expect: dict) -> list[str]:
    """Why a C2 checkpoint is not usable for the vision context (empty list = usable).

    expect: {"label_scheme": "daisee_binary", "cell": "gru", "window": 20, "target": "engagement"}"""
    probs = []
    if meta.get("model_kind") != "temporal":
        probs.append(f"model_kind is {meta.get('model_kind')!r}, not 'temporal' (the GRU)")
    if meta.get("demo_only") or meta.get("data_source") != "real":
        probs.append("DEMO ONLY model trained on synthetic data (never used by Component 3)")
    if meta.get("feature_schema_version") != C2_SCHEMA_VERSION:
        probs.append(f"feature schema {meta.get('feature_schema_version')!r} != {C2_SCHEMA_VERSION}")
    if list(meta.get("feature_columns", [])) != C2_FEATURE_COLUMNS:
        probs.append("feature columns differ from C2 schema 1.0.0 (15 features)")
    ls = meta.get("label_scheme") or {}
    if expect.get("label_scheme") and ls.get("name") != expect["label_scheme"]:
        probs.append(f"label scheme is {ls.get('name')!r}; the team decided {expect['label_scheme']!r} "
                     "(engagement low = 0-1, high = 2-3)")
    if expect.get("target") and ls.get("target") not in (None, expect["target"]):
        probs.append(f"label target is {ls.get('target')!r}, not {expect['target']!r}")
    mp = meta.get("model_params") or {}
    if expect.get("cell") and mp.get("cell") != expect["cell"]:
        probs.append(f"cell is {mp.get('cell')!r}, not {expect['cell']!r}")
    if expect.get("window") and int(meta.get("window", -1)) != int(expect["window"]):
        probs.append(f"window is {meta.get('window')} frames, not {expect['window']}")
    norm = meta.get("normalizer") or {}
    if not all(k in norm for k in ("method", "fill_values", "a", "b")) or len(norm.get("a", [])) != len(C2_FEATURE_COLUMNS):
        probs.append("normaliser parameters missing or wrong length in metadata.json")
    return probs


def find_model_dirs(models_dir: Path | str) -> list[Path]:
    """Every C2 model folder (has metadata.json), newest first by created_utc (as C2 registry.list_models)."""
    root = Path(models_dir)
    if not root.exists():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists()]

    def created(d):
        try:
            return read_metadata(d).get("created_utc", "")
        except Exception:
            return ""
    return sorted(dirs, key=created, reverse=True)


def select_model_dir(models_dir: Path | str, model_name: str | None, expect: dict) -> tuple[Path | None, dict]:
    """The model folder to explain: `model_name`, or (auto) the newest usable one.
    Returns (folder or None, {folder name: [problems]} for every candidate looked at)."""
    report = {}
    cands = [Path(models_dir) / model_name] if model_name and model_name != "auto" else find_model_dirs(models_dir)
    for d in cands:
        try:
            probs = metadata_problems(read_metadata(d), expect)
        except VisionModelError as e:
            probs = [str(e)]
        if not (d / "model.pt").exists():
            probs.append("model.pt (weights) missing")
        report[d.name] = probs
        if not probs:
            return d, report
    return None, report


@dataclass
class LoadedVisionModel:
    model: TemporalClassifier          # eval mode
    normalizer: Normalizer
    meta: dict
    class_names: list[str]
    path: Path
    weights: str = "trained"           # trained | randomised


def load_vision_model(model_dir: Path | str) -> LoadedVisionModel:
    model_dir = Path(model_dir)
    meta = read_metadata(model_dir)
    model = TemporalClassifier(**meta["model_params"])
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return LoadedVisionModel(model=model.eval(), normalizer=Normalizer.from_dict(meta["normalizer"]), meta=meta,
                             class_names=list(meta["label_scheme"]["class_names"]), path=model_dir)


def randomise_vision(lm: LoadedVisionModel, seed: int) -> LoadedVisionModel:
    """Same architecture, freshly initialised weights (negative control). Normaliser kept; original untouched."""
    torch.manual_seed(seed)
    model = TemporalClassifier(**lm.meta["model_params"]).eval()
    return LoadedVisionModel(model=model, normalizer=lm.normalizer, meta=dict(lm.meta), class_names=list(lm.class_names),
                             path=lm.path, weights="randomised")
