"""Extract and cache per-frame features for real dataset clips.

Output: data/processed/features/<clip_id>.csv.gz  (+ features_index.csv)
A cached file is reused only if its schema version and extraction settings match.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.data.feature_schema import ALL_COLUMNS, SCHEMA_VERSION
from src.data.video_loader import iter_sampled_frames
from src.utils.config import resolve_path
from src.utils.logging_config import get_logger
from src.vision.feature_extractor import FrameFeatureExtractor

log = get_logger("extract")


def _settings_key(cfg: dict, backend: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "target_fps": cfg["extraction"]["target_fps"],
            "max_frames": cfg["extraction"]["max_frames_per_video"], "backend": backend,
            "ear_threshold": cfg["vision"]["ear_blink_threshold"]}


def extract_clip(path: Path, extractor: FrameFeatureExtractor, cfg: dict, labels: dict) -> pd.DataFrame:
    rows = []
    for idx, ts, frame in iter_sampled_frames(path, cfg["extraction"]["target_fps"],
                                              cfg["extraction"]["max_frames_per_video"]):
        row, _ = extractor.extract(frame, ts, idx, labels=labels)
        rows.append(row)
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def run_extraction(cfg: dict, manifest: pd.DataFrame, detector=None, limit: int | None = None,
                   overwrite: bool = False, source: str = "daisee") -> pd.DataFrame:
    """manifest columns: clip_id, video_path, participant_id, split, label_* (from the audit)."""
    out_dir = resolve_path(cfg, "features_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrameFeatureExtractor(cfg, detector=detector, source=source)
    if extractor.backend in ("none", "opencv_haar"):
        log.warning("Face backend '%s' provides no landmarks; landmark features will be NaN. "
                    "Install a MediaPipe face-landmark backend before real extraction.", extractor.backend)
    settings = _settings_key(cfg, extractor.backend)
    index_rows = []
    todo = manifest if limit is None else manifest.head(limit)
    try:
        for i, rec in enumerate(todo.itertuples(index=False), 1):
            clip = str(rec.clip_id)
            target = out_dir / f"{clip}.csv.gz"
            sidecar = out_dir / f"{clip}.json"
            if cfg["extraction"].get("cache", True) and not overwrite and target.exists() and sidecar.exists():
                with open(sidecar, encoding="utf-8") as fh:
                    if json.load(fh).get("settings") == settings:
                        index_rows.append({"clip_id": clip, "path": str(target), "status": "cached"})
                        continue
            labels = {k: getattr(rec, f"label_{k}") for k in ("engagement", "boredom", "confusion", "frustration")}
            extractor.reset(participant_id=str(rec.participant_id), video_id=clip, split=str(rec.split))
            t0 = time.perf_counter()
            try:
                df = extract_clip(Path(rec.video_path), extractor, cfg, labels)
            except Exception as exc:
                log.error("Clip %s failed: %s", clip, exc)
                index_rows.append({"clip_id": clip, "path": None, "status": f"error: {exc}"})
                continue
            df.to_csv(target, index=False, compression="gzip")
            with open(sidecar, "w", encoding="utf-8") as fh:
                json.dump({"settings": settings, "n_rows": len(df),
                           "quality_ok_ratio": float(df["quality_ok"].mean()) if len(df) else 0.0,
                           "seconds": round(time.perf_counter() - t0, 2)}, fh, indent=2)
            index_rows.append({"clip_id": clip, "path": str(target), "status": "extracted", "rows": len(df)})
            log.info("[%d/%d] %s: %d frames", i, len(todo), clip, len(df))
    finally:
        extractor.close()
    index = pd.DataFrame(index_rows)
    index.to_csv(out_dir / "features_index.csv", index=False)
    return index


def load_feature_table(cfg: dict, clip_ids: list[str] | None = None) -> pd.DataFrame:
    out_dir = resolve_path(cfg, "features_dir")
    files = sorted(out_dir.glob("*.csv.gz"))
    if clip_ids is not None:
        wanted = set(map(str, clip_ids))
        files = [f for f in files if f.name[:-len(".csv.gz")] in wanted]
    if not files:
        return pd.DataFrame(columns=ALL_COLUMNS)
    return pd.concat([pd.read_csv(f, dtype={"participant_id": str, "video_id": str}) for f in files],
                     ignore_index=True)
