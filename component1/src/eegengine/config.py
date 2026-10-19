"""Small helpers for YAML configs, project paths and seeding."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def config_path(path: Path | str) -> Path:
    """Find a config file: as given, relative to the project root, or relative to configs/."""
    p = Path(path)
    for cand in (p, PROJECT_ROOT / p, CONFIG_DIR / p):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Config not found: {path}")


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config (see `config_path` for how the file is located)."""
    with open(config_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(path: Path | str) -> Path:
    """Resolve a path from a config relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int = 42) -> None:
    """Fix Python and NumPy seeds (PyTorch seeding is added where torch is used)."""
    random.seed(seed)
    np.random.seed(seed)
