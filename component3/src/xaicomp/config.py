"""YAML configs of Component 3 (configs/) and read-only access to Component 1's configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from xaicomp.paths import C1_CONFIG_DIR, CONFIG_DIR


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_c3_config(name: str) -> dict[str, Any]:
    """Load configs/<name> (e.g. 'run.yaml' or 'contexts/eeg_T1_A8.yaml')."""
    return load_yaml(CONFIG_DIR / name)


def load_c1_config(name: str) -> dict[str, Any]:
    """Load one of Component 1's configs READ-ONLY (e.g. 'preprocess.yaml', 'experiments/deep.yaml')."""
    return load_yaml(C1_CONFIG_DIR / name)


def deep_merge(base: dict, override: dict | None) -> dict:
    """Recursive dict merge; `override` wins. Returns a new dict."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_context(name_or_path: str) -> dict[str, Any]:
    """Load a context YAML by file name ('eeg_T1_A8') or path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / "contexts" / f"{name_or_path}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / "contexts" / p
    return load_yaml(p)


def run_settings(pilot: bool) -> dict[str, Any]:
    """Merged settings for one run: run.yaml 'full' budget, overridden by 'pilot' when pilot=True,
    plus the explainer / metric / stats / ranking configs."""
    run = load_c3_config("run.yaml")
    budget = deep_merge(run["full"], run["pilot"]) if pilot else copy.deepcopy(run["full"])
    return {
        "pilot": pilot,
        "budget": budget,
        "run": run,
        "explainers": load_c3_config("explainers.yaml"),
        "metrics": load_c3_config("metrics.yaml"),
        "stats": load_c3_config("stats.yaml"),
        "ranking": load_c3_config("ranking.yaml"),
    }
