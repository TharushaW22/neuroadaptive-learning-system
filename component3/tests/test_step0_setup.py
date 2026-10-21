"""Step 0: environment, read-only Component 1 import, write guard, git safety."""

import subprocess
import sys

import pytest

from xaicomp._bootstrap import C1_ROOT, C3_ROOT, REPO_ROOT
from xaicomp.paths import OutsideComponent3Error, safe_path


def test_c1_predictor_loads_with_expected_facts():
    from eegengine.inference.predictors import load_predictor
    p = load_predictor("T1_A8_cnn_lstm_trans")
    assert sum(x.numel() for x in p.model.parameters()) == 130_616
    assert p.temperature == pytest.approx(1.101, abs=1e-3)
    assert p.classes == ["low", "high"]


def test_bytecode_never_written_into_component1():
    assert sys.pycache_prefix is not None
    assert str(C3_ROOT) in sys.pycache_prefix


def test_c1_not_installed_in_venv():
    import importlib.metadata as md
    with pytest.raises(md.PackageNotFoundError):
        md.version("eegengine")


def test_safe_path_refuses_outside_component3(tmp_path):
    with pytest.raises(OutsideComponent3Error):
        safe_path(C1_ROOT / "results" / "x.json", mkdir_parent=False)
    with pytest.raises(OutsideComponent3Error):
        safe_path(REPO_ROOT / "README.md", mkdir_parent=False)
    assert safe_path("results/_scratch/ok.txt", mkdir_parent=False).is_relative_to(C3_ROOT)


def test_git_status_only_component3_changed():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git not available")
    paths = [line[3:].strip().strip('"') for line in out.stdout.splitlines() if line.strip()]
    outside = [p for p in paths if not p.startswith("component3/")]
    assert outside == [], f"changes outside component3/: {outside}"
