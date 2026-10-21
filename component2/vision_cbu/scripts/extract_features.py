"""Extract and cache per-frame features for every included DAiSEE clip.

    python scripts/extract_features.py                # all clips (cached ones skipped)
    python scripts/extract_features.py --limit 20     # quick trial
    python scripts/extract_features.py --overwrite    # ignore cache
Refuses (exit 2) when the dataset is not available.
"""
import argparse
import sys

from _bootstrap import EXIT_OK, EXIT_REFUSED
from src.data.extract_features import run_extraction
from src.data.validate_dataset import RealDataUnavailableError, get_clip_manifest
from src.utils.config import load_config
from src.utils.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--splits", nargs="*", default=None, help="e.g. --splits train validation")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    try:
        manifest = get_clip_manifest(cfg, probe=False)
    except RealDataUnavailableError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    if a.splits:
        manifest = manifest[manifest["split"].isin(a.splits)]
    print(f"Extracting features for {len(manifest) if a.limit is None else min(a.limit, len(manifest))} clips...")
    idx = run_extraction(cfg, manifest, limit=a.limit, overwrite=a.overwrite)
    print(idx["status"].str.split(":").str[0].value_counts().to_string())
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
