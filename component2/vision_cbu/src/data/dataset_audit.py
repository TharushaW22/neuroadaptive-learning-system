"""Dataset audit: inspects whatever is actually in data/raw/daisee and reports it.

* Runs on an empty folder (status DATASET_NOT_AVAILABLE, exit code 0).
* Never downloads, never invents metadata or labels.
* Writes outputs/audit/audit_report.json and outputs/audit/audit_clips.csv.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src.data.annotation_parser import LABEL_CANDIDATES, load_all_annotations
from src.data.video_loader import (find_duplicate_stems, index_videos, participant_from_path,
                                   probe_video, split_from_path)
from src.utils.config import resolve_path

STATUS_NOT_AVAILABLE = "DATASET_NOT_AVAILABLE"
STATUS_PARTIAL = "DATASET_PARTIAL"
STATUS_READY = "DATASET_READY"


def run_audit(cfg: dict, probe: bool = True, max_probe: int | None = None, write: bool = True) -> dict:
    videos_dir = resolve_path(cfg, "daisee_videos")
    ann_dir = resolve_path(cfg, "daisee_annotations")
    exts = tuple(cfg["extraction"]["video_extensions"])

    video_index = index_videos(videos_dir, exts)
    annotations, parsed_files = load_all_annotations(ann_dir)

    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "DAiSEE",
        "videos_dir": str(videos_dir),
        "annotations_dir": str(ann_dir),
        "n_video_files": len(video_index),
        "n_annotation_files": len(parsed_files),
        "annotation_files": [],
        "messages": [],
    }
    for p in parsed_files:
        report["annotation_files"].append({
            "file": p.path.name, "inferred_split": p.split, "raw_columns": p.raw_columns,
            "column_map": p.column_map, "unmatched_columns": p.unmatched_columns,
            "valid_rows": int(len(p.table)), "invalid_rows": len(p.invalid_rows), "errors": p.errors,
        })

    if not video_index and not parsed_files:
        report["status"] = STATUS_NOT_AVAILABLE
        report["messages"].append(
            "No DAiSEE videos or annotation CSVs found. This is expected before the dataset is "
            "downloaded. See data/raw/daisee/README.md for where to place the files. "
            "Demo mode and all automated tests work without the dataset.")
        if write:
            _write(cfg, report, pd.DataFrame())
        return report
    if not video_index:
        report["messages"].append("Annotation files found but no video files.")
    if not parsed_files:
        report["messages"].append("Video files found but no annotation CSVs.")

    invalid_rows = [r for p in parsed_files for r in p.invalid_rows]
    report["invalid_annotation_rows"] = invalid_rows
    report["duplicate_video_stems"] = find_duplicate_stems(videos_dir, exts)

    # Clip-level table: union of annotated clips and video files
    ann = annotations.set_index("clip_id") if not annotations.empty else pd.DataFrame()
    clip_ids = sorted(set(video_index) | set(ann.index if not ann.empty else []))
    rows = []
    probed = 0
    for cid in clip_ids:
        path = video_index.get(cid)
        row = {"clip_id": cid, "video_path": str(path) if path else None,
               "annotation_split": None, "folder_split": split_from_path(path) if path else None,
               "participant_id": participant_from_path(path, videos_dir) if path else None,
               "readable": None, "frame_count": None, "fps": None, "width": None, "height": None,
               "duration_s": None}
        for lab in LABEL_CANDIDATES:
            row[f"label_{lab}"] = None
        if not ann.empty and cid in ann.index:
            rec = ann.loc[cid]
            row["annotation_split"] = rec["split"]
            for lab in LABEL_CANDIDATES:
                v = rec.get(lab)
                row[f"label_{lab}"] = None if pd.isna(v) else int(v)
        if path and probe and (max_probe is None or probed < max_probe):
            meta = probe_video(path)
            probed += 1
            row.update({k: meta[k] for k in ("readable", "frame_count", "fps", "width", "height", "duration_s")})
            row["probe_error"] = meta["error"]
        reasons = []
        if not path:
            reasons.append("video file missing")
        if row["label_engagement"] is None:
            reasons.append("engagement label missing")
        if row["readable"] is False:
            reasons.append(f"unreadable video ({row.get('probe_error')})")
        if path and row["participant_id"] is None:
            reasons.append("participant id not derivable from folder layout")
        split = row["annotation_split"] if row["annotation_split"] not in (None, "unspecified") else row["folder_split"]
        if row["annotation_split"] not in (None, "unspecified") and row["folder_split"] and \
                row["annotation_split"] != row["folder_split"]:
            reasons.append(f"split conflict: annotation={row['annotation_split']} folder={row['folder_split']}")
        row["split"] = split or "unassigned"
        row["included"] = not reasons
        row["exclusion_reason"] = "; ".join(reasons)
        rows.append(row)
    clips = pd.DataFrame(rows)

    inc = clips[clips["included"]] if not clips.empty else clips
    report["n_clips_total"] = int(len(clips))
    report["n_clips_included"] = int(len(inc))
    report["n_clips_excluded"] = int(len(clips) - len(inc))
    report["exclusion_reasons"] = dict(Counter(
        r for rs in clips.loc[~clips["included"], "exclusion_reason"] for r in rs.split("; ") if r)) if len(clips) else {}
    report["n_unreadable_videos"] = int((clips["readable"] == False).sum()) if len(clips) else 0  # noqa: E712
    report["videos_probed"] = probed
    report["split_counts"] = inc["split"].value_counts().to_dict() if len(inc) else {}
    report["class_distribution"] = {}
    for lab in LABEL_CANDIDATES:
        col = f"label_{lab}"
        if len(inc):
            dist = inc.groupby("split")[col].value_counts().unstack(fill_value=0)
            report["class_distribution"][lab] = {s: {str(int(k)): int(v) for k, v in r.items()}
                                                 for s, r in dist.iterrows()}
    report["participants"] = _participant_summary(inc)
    report["leakage"] = _leakage(clips, annotations)

    ready = len(inc) > 0 and {"train", "test"} <= set(report["split_counts"])
    report["status"] = STATUS_READY if ready else STATUS_PARTIAL
    if not ready:
        report["messages"].append("Dataset incomplete: need included clips in at least train and test splits.")
    if report["leakage"]["participant_overlap"]:
        report["messages"].append("WARNING: participants appear in more than one split (see leakage).")
    if write:
        _write(cfg, report, clips)
    return report


def _participant_summary(inc: pd.DataFrame) -> dict:
    if inc.empty:
        return {"n_participants": 0}
    known = inc[inc["participant_id"].notna()]
    return {
        "n_participants": int(known["participant_id"].nunique()),
        "per_split": known.groupby("split")["participant_id"].nunique().to_dict(),
        "clips_per_participant_median": float(known.groupby("participant_id").size().median()) if len(known) else None,
    }


def _leakage(clips: pd.DataFrame, annotations: pd.DataFrame) -> dict:
    out = {"participant_overlap": {}, "clips_in_multiple_annotation_splits": []}
    if not annotations.empty:
        multi = annotations.groupby("clip_id")["split"].nunique()
        out["clips_in_multiple_annotation_splits"] = sorted(multi[multi > 1].index.tolist())
    if clips.empty:
        return out
    known = clips[clips["participant_id"].notna() & clips["included"]]
    by_p = known.groupby("participant_id")["split"].apply(lambda s: sorted(set(s)))
    out["participant_overlap"] = {p: s for p, s in by_p.items() if len(s) > 1}
    return out


def _write(cfg: dict, report: dict, clips: pd.DataFrame) -> None:
    out_dir = resolve_path(cfg, "audit_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    clips.to_csv(out_dir / "audit_clips.csv", index=False)


def load_latest_audit(cfg: dict) -> tuple[dict | None, pd.DataFrame | None]:
    out_dir = resolve_path(cfg, "audit_dir")
    rp, cp = out_dir / "audit_report.json", out_dir / "audit_clips.csv"
    if not rp.exists():
        return None, None
    with open(rp, encoding="utf-8") as fh:
        rep = json.load(fh)
    clips = pd.DataFrame()
    if cp.exists():
        try:
            clips = pd.read_csv(cp, dtype={"clip_id": str, "participant_id": str})
        except pd.errors.EmptyDataError:   # empty audit (no dataset); on Windows the file is just "\r\n"
            clips = pd.DataFrame()
    return rep, clips
