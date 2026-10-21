"""Project status used by the dashboard and CLI (cheap: no video probing)."""
from __future__ import annotations

from src.data.annotation_parser import discover_annotation_files
from src.data.dataset_audit import load_latest_audit
from src.data.video_loader import index_videos
from src.models.inference import find_validated_model
from src.models.registry import list_models
from src.utils.config import resolve_path

DEMO_MODE = "DEMO MODE"
RESEARCH_MODE = "RESEARCH MODE"


def get_status(cfg: dict) -> dict:
    n_videos = len(index_videos(resolve_path(cfg, "daisee_videos"), cfg["extraction"]["video_extensions"]))
    n_ann = len(discover_annotation_files(resolve_path(cfg, "daisee_annotations")))
    feats_dir = resolve_path(cfg, "features_dir")
    n_feat = len(list(feats_dir.glob("*.csv.gz"))) if feats_dir.exists() else 0
    audit, _ = load_latest_audit(cfg)
    real_models = [m for m in list_models(resolve_path(cfg, "models_dir")) if m.get("data_source") == "real"]
    demo_models = list_models(resolve_path(cfg, "demo_dir") / "models")
    validated = find_validated_model(cfg)
    dataset_present = n_videos > 0 and n_ann > 0
    forced = cfg.get("dashboard", {}).get("mode", "auto")
    mode = RESEARCH_MODE if (dataset_present and forced != "demo") else DEMO_MODE
    return {
        "mode": mode, "dataset_present": dataset_present, "n_videos": n_videos, "n_annotation_files": n_ann,
        "audit_status": audit.get("status") if audit else "NOT RUN",
        "audit_generated": audit.get("generated_utc") if audit else None,
        "n_feature_files": n_feat, "real_models": real_models, "demo_models": demo_models,
        "validated_model": validated,
        "demo_data_present": (resolve_path(cfg, "demo_dir") / "synthetic_features.csv").exists(),
    }
