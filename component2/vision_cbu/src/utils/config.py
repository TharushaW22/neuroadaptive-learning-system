"""Configuration loading, path resolution and reproducibility helpers."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Load the default config, deep-merge an optional experiment config and overrides."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Return an absolute Path for cfg['paths'][key] (relative paths are root-relative)."""
    raw = cfg["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(cfg.get("_project_root", PROJECT_ROOT)) / p
    return p


def with_root(cfg: dict, root: str | os.PathLike) -> dict:
    """Copy of cfg whose relative paths resolve against another root (used by tests)."""
    new = copy.deepcopy(cfg)
    new["_project_root"] = str(root)
    return new


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
