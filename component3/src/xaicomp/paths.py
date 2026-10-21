"""Project paths and the write guard.

Every function in xaicomp that writes a file calls `safe_path()` first. It refuses any path
outside component3/, which enforces safety rule A1 in code, not only in documentation.
"""

from __future__ import annotations

from pathlib import Path

from xaicomp._bootstrap import C1_ROOT, C3_ROOT

CONFIG_DIR = C3_ROOT / "configs"
DATA_DIR = C3_ROOT / "data"
RAW_STEW_DIR = DATA_DIR / "raw" / "STEW"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = C3_ROOT / "artifacts"
RESULTS_DIR = C3_ROOT / "results"

C1_CONFIG_DIR = C1_ROOT / "configs"
C1_MODELS_DIR = C1_ROOT / "models"
C1_RESULTS_DIR = C1_ROOT / "results"


class OutsideComponent3Error(PermissionError):
    """Raised when code tries to write outside component3/."""


def resolve_c3(path: Path | str) -> Path:
    """Resolve a (possibly relative) path against component3/."""
    p = Path(path)
    return (p if p.is_absolute() else C3_ROOT / p).resolve()


def safe_path(path: Path | str, mkdir_parent: bool = True) -> Path:
    """Return the resolved path if it lies inside component3/, else raise.

    Use for EVERY file that is written. Optionally creates the parent folder.
    """
    p = resolve_c3(path)
    root = C3_ROOT.resolve()
    if p != root and root not in p.parents:
        raise OutsideComponent3Error(f"refusing to write outside component3/: {p}")
    if mkdir_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
