# Dataset guide - adding DAiSEE later

## Steps
1. Request/download DAiSEE from https://people.iith.ac.in/vineethnb/resources/daisee/ and read its licence.
2. Copy the video folder (official structure) to `data/raw/daisee/videos/`.
3. Copy the official label CSVs to `data/raw/daisee/annotations/`.
4. `python scripts\audit_dataset.py` - read the output and `outputs/audit/audit_report.json`.
5. Fix anything the audit flags (missing videos, unreadable files, leakage) **without editing labels**.
6. `python scripts\extract_features.py --limit 20` (trial), then without `--limit`.
7. Train and evaluate (README steps 11-13).

## What the audit checks
| Check | Output |
|---|---|
| Video files found (recursive, `.avi/.mp4/.mov/.mkv`) | `n_video_files`, duplicate file stems |
| Annotation CSVs and **discovered** column mapping | `annotation_files[*].column_map`, unmatched columns |
| Labels in 0..3, non-numeric, missing, duplicates | `invalid_annotation_rows` |
| Video metadata from the container (frames, fps, size, duration) | per-clip CSV |
| Unreadable / undecodable videos | `n_unreadable_videos`, exclusion reason |
| Clip with label but no video, or video without label | exclusion reason |
| Split (from label file name and from folder) and conflicts | `split`, exclusion reason |
| Participant ID from `<subject>` folder | `participants`, exclusion if not derivable |
| Participant in more than one split; clip in multiple label files | `leakage` |
| Class distribution per split for all four labels | `class_distribution` |

Outputs: `outputs/audit/audit_report.json` (summary) and `outputs/audit/audit_clips.csv`
(one row per clip with `included` and `exclusion_reason`). Status is
`DATASET_NOT_AVAILABLE`, `DATASET_PARTIAL` or `DATASET_READY` (included clips in train and test).

## Feature cache
`data/processed/features/<clipID>.csv.gz` + `<clipID>.json` (settings, rows, quality ratio,
extraction time) and `features_index.csv`. A cache file is reused only if schema version, target
fps, frame limit, face backend and EAR threshold are unchanged; otherwise use `--overwrite`.
Windowed sequences can be saved with `SequenceSet.save()` to `data/processed/sequences/`.

## Practical notes
* Extraction is CPU-bound: start with `--limit`, check `quality_ok_ratio` in the sidecar JSONs.
* Use `--splits train validation` to extract in stages.
* Keep the face backend constant for an entire experiment (it is recorded in the cache key).

## Local modifications log
Record every deviation from the official release here (date, files, change, reason, who).

| Date | Files | Change | Reason | By |
|---|---|---|---|---|
| | | | | |
