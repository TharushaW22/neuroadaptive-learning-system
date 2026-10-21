"""Audit data/raw/daisee. Exits 0 even when the dataset is absent (that is expected).

    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --no-probe          # skip opening every video (fast)
    python scripts/audit_dataset.py --max-probe 200     # open at most 200 videos
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from src.data.dataset_audit import STATUS_NOT_AVAILABLE, run_audit
from src.utils.config import load_config, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--max-probe", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rep = run_audit(cfg, probe=not a.no_probe, max_probe=a.max_probe)
    print(f"Status: {rep['status']}")
    for m in rep["messages"]:
        print(f"  - {m}")
    if rep["status"] != STATUS_NOT_AVAILABLE:
        keys = ["n_video_files", "n_annotation_files", "n_clips_total", "n_clips_included", "n_clips_excluded",
                "n_unreadable_videos", "split_counts", "exclusion_reasons", "participants"]
        print(json.dumps({k: rep.get(k) for k in keys}, indent=2, default=str))
        for f in rep["annotation_files"]:
            print(f"  annotation {f['file']}: split={f['inferred_split']} map={f['column_map']} "
                  f"valid={f['valid_rows']} invalid={f['invalid_rows']} errors={f['errors']}")
        if rep["leakage"]["participant_overlap"]:
            print(f"  LEAKAGE WARNING: {len(rep['leakage']['participant_overlap'])} participant(s) in >1 split")
    print(f"Reports: {resolve_path(cfg, 'audit_dir')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
