"""Discover and parse DAiSEE-style annotation CSVs WITHOUT assuming exact column names.

Column names are normalised (strip, lower-case, remove non-alphanumerics) and matched
against candidate names. Anything that cannot be matched is reported, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CLIP_ID_CANDIDATES = ["clipid", "clip", "videoid", "video", "filename", "file", "clipname"]
LABEL_CANDIDATES = {
    "engagement": ["engagement", "engaged"],
    "boredom": ["boredom", "bored"],
    "confusion": ["confusion", "confused"],
    "frustration": ["frustration", "frustrated"],
}
SPLIT_PATTERNS = [("train", r"train"), ("validation", r"valid|val\b|validation"), ("test", r"test")]
VALID_LEVELS = {0, 1, 2, 3}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def infer_split_from_filename(path: Path) -> str | None:
    stem = path.stem.lower()
    if stem.startswith("all"):
        return None  # e.g. AllLabels.csv: union file, no split information
    for split, pattern in SPLIT_PATTERNS:
        if re.search(pattern, stem):
            return split
    return None


def clip_key(clip_id: str) -> str:
    """Canonical clip key: file stem without directory or extension (e.g. '1100011002')."""
    s = str(clip_id).strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"\.(avi|mp4|mov|mkv)$", "", s, flags=re.IGNORECASE)


@dataclass
class ParsedAnnotations:
    path: Path
    split: str | None
    raw_columns: list[str]
    column_map: dict[str, str]          # canonical -> raw column
    unmatched_columns: list[str]
    table: pd.DataFrame                  # canonical columns, valid rows only
    invalid_rows: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def discover_annotation_files(annotation_dir: Path) -> list[Path]:
    if not annotation_dir.exists():
        return []
    return sorted(p for p in annotation_dir.rglob("*") if p.suffix.lower() == ".csv" and p.is_file())


def _match_columns(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    norm = {normalise(c): c for c in columns}
    mapping: dict[str, str] = {}
    for cand in CLIP_ID_CANDIDATES:
        if cand in norm:
            mapping["clip_id"] = norm[cand]
            break
    for canonical, cands in LABEL_CANDIDATES.items():
        for cand in cands:
            if cand in norm:
                mapping[canonical] = norm[cand]
                break
    used = set(mapping.values())
    return mapping, [c for c in columns if c not in used]


def parse_annotation_file(path: Path) -> ParsedAnnotations:
    split = infer_split_from_filename(path)
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # unreadable / malformed file
        return ParsedAnnotations(path, split, [], {}, [], pd.DataFrame(), errors=[f"Unreadable CSV: {exc}"])

    mapping, unmatched = _match_columns(list(raw.columns))
    errors: list[str] = []
    if "clip_id" not in mapping:
        errors.append(f"No clip-identifier column found among {list(raw.columns)}")
    label_cols = [k for k in LABEL_CANDIDATES if k in mapping]
    if not label_cols:
        errors.append(f"No recognised label columns among {list(raw.columns)}")
    if errors:
        return ParsedAnnotations(path, split, list(raw.columns), mapping, unmatched, pd.DataFrame(), errors=errors)

    rows, invalid = [], []
    for i, rec in raw.iterrows():
        cid = str(rec[mapping["clip_id"]]).strip()
        reasons = []
        if not cid:
            reasons.append("empty clip id")
        parsed = {"clip_id": clip_key(cid) if cid else "", "raw_clip_id": cid}
        for lab in LABEL_CANDIDATES:
            if lab not in mapping:
                parsed[lab] = pd.NA
                continue
            val = str(rec[mapping[lab]]).strip()
            try:
                f = float(val)
                if f != int(f) or int(f) not in VALID_LEVELS:
                    reasons.append(f"{lab} out of range: {val!r}")
                parsed[lab] = int(f)
            except ValueError:
                reasons.append(f"{lab} missing/non-numeric: {val!r}")
                parsed[lab] = pd.NA
        if reasons:
            invalid.append({"file": path.name, "row": int(i) + 2, "clip_id": cid, "reasons": "; ".join(reasons)})
        else:
            rows.append(parsed)

    table = pd.DataFrame(rows, columns=["clip_id", "raw_clip_id", *LABEL_CANDIDATES.keys()])
    table["split"] = split if split else "unspecified"
    table["annotation_file"] = path.name
    # duplicates inside one file
    dups = table[table.duplicated("clip_id", keep=False)]
    for cid in sorted(set(dups["clip_id"])):
        invalid.append({"file": path.name, "row": -1, "clip_id": cid, "reasons": "duplicate clip id within file"})
    table = table.drop_duplicates("clip_id", keep=False).reset_index(drop=True)
    return ParsedAnnotations(path, split, list(raw.columns), mapping, unmatched, table, invalid, [])


def load_all_annotations(annotation_dir: Path) -> tuple[pd.DataFrame, list[ParsedAnnotations]]:
    """Parse every CSV. Split-specific files take precedence over union files (e.g. AllLabels)."""
    parsed = [parse_annotation_file(p) for p in discover_annotation_files(annotation_dir)]
    good = [p for p in parsed if p.ok and not p.table.empty]
    split_tables = [p.table for p in good if p.split]
    union_tables = [p.table for p in good if not p.split]
    combined = pd.concat(split_tables, ignore_index=True) if split_tables else pd.DataFrame()
    if union_tables:
        union = pd.concat(union_tables, ignore_index=True).drop_duplicates("clip_id")
        if combined.empty:
            combined = union
        else:
            extra = union[~union["clip_id"].isin(combined["clip_id"])]
            combined = pd.concat([combined, extra], ignore_index=True)
    return combined, parsed
