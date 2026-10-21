import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.annotation_parser import clip_key, infer_split_from_filename, parse_annotation_file
from src.data.dataset_audit import STATUS_NOT_AVAILABLE, STATUS_PARTIAL, STATUS_READY, run_audit
from src.data.validate_dataset import RealDataUnavailableError, assert_no_split_leakage, get_clip_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_audit_empty_dataset_succeeds_and_writes_reports(cfg, tmp_path):
    rep = run_audit(cfg)
    assert rep["status"] == STATUS_NOT_AVAILABLE
    assert rep["n_video_files"] == 0 and rep["n_annotation_files"] == 0
    assert (tmp_path / "outputs/audit/audit_report.json").exists()
    assert (tmp_path / "outputs/audit/audit_clips.csv").exists()


def test_audit_script_exits_zero_without_dataset():
    r = subprocess.run([sys.executable, "scripts/audit_dataset.py", "--no-probe"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "Status:" in r.stdout


def test_manifest_refuses_without_dataset(cfg):
    with pytest.raises(RealDataUnavailableError):
        get_clip_manifest(cfg)


def test_annotation_parsing_discovers_columns(tmp_path):
    p = tmp_path / "TrainLabels.csv"
    p.write_text("ClipID,Boredom,Engagement,Confusion,Frustration \n1100011002.avi,0,2,0,0\n"
                 "1100011003.avi,1,9,0,0\n1100011004.avi,0,,0,0\n", encoding="utf-8")
    parsed = parse_annotation_file(p)
    assert parsed.ok and parsed.split == "train"
    assert parsed.column_map["frustration"] == "Frustration "      # trailing space discovered, not assumed
    assert list(parsed.table["clip_id"]) == ["1100011002"]
    assert len(parsed.invalid_rows) == 2


def test_annotation_unknown_columns_reported(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    parsed = parse_annotation_file(p)
    assert not parsed.ok and parsed.errors


def test_split_and_clip_helpers():
    assert infer_split_from_filename(Path("ValidationLabels.csv")) == "validation"
    assert infer_split_from_filename(Path("AllLabels.csv")) is None
    assert clip_key("DataSet/Train/110001/1100011002/1100011002.avi") == "1100011002"


def test_audit_with_fake_dataset(cfg, fake_daisee):
    rep = run_audit(cfg)
    assert rep["status"] == STATUS_READY
    assert rep["n_clips_included"] == 14
    assert rep["participants"]["n_participants"] == 7
    assert rep["leakage"]["participant_overlap"] == {}
    clips = get_clip_manifest(cfg)
    assert set(clips["split"]) == {"train", "validation", "test"}


def test_audit_records_missing_video_and_unreadable(cfg, fake_daisee):
    ann = fake_daisee["files"]["Test"]
    ann.write_text(ann.read_text() + "9999990000.avi,0,1,0,0\n", encoding="utf-8")
    bad = fake_daisee["videos"] / "Test" / "310001" / "3100010099" / "3100010099.avi"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a video")
    rep = run_audit(cfg)
    clips = pd.read_csv(Path(cfg["_project_root"]) / "outputs/audit/audit_clips.csv", dtype={"clip_id": str})
    reasons = dict(zip(clips["clip_id"], clips["exclusion_reason"].fillna("")))
    assert "video file missing" in reasons["9999990000"]
    assert "unreadable" in reasons["3100010099"]
    assert rep["n_unreadable_videos"] == 1


def test_audit_detects_participant_leakage(cfg, fake_daisee):
    from tests.conftest import write_video
    clip = "1100010007"
    write_video(fake_daisee["videos"] / "Test" / "110001" / clip / f"{clip}.avi")
    ann = fake_daisee["files"]["Test"]
    ann.write_text(ann.read_text() + f"{clip}.avi,0,1,0,0\n", encoding="utf-8")
    rep = run_audit(cfg)
    assert "110001" in rep["leakage"]["participant_overlap"]


def test_partial_dataset(cfg, fake_daisee):
    for f in fake_daisee["annotations"].glob("*.csv"):
        f.unlink()
    rep = run_audit(cfg, probe=False)
    assert rep["status"] == STATUS_PARTIAL


def test_leakage_assertion():
    df = pd.DataFrame({"participant_id": ["a", "a", "b"], "video_id": ["v1", "v2", "v3"], "split": ["train", "test", "test"]})
    with pytest.raises(ValueError):
        assert_no_split_leakage(df, group_cols=("participant_id",))
    assert_no_split_leakage(df, group_cols=("video_id",))


def test_load_latest_audit_handles_empty_clip_file_windows_newline(cfg, tmp_path):
    from src.data.dataset_audit import load_latest_audit
    run_audit(cfg)
    (tmp_path / "outputs/audit/audit_clips.csv").write_bytes(b"\r\n")   # what Windows writes for an empty table
    rep, clips = load_latest_audit(cfg)
    assert rep["status"] == STATUS_NOT_AVAILABLE and clips.empty
