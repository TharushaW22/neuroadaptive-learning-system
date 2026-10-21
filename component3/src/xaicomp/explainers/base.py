"""The common output of every explainer, so all comparisons are like-for-like.

An Explanation is one attribution vector over the SAME interpretable features (here 84 channel x
band components) for one instance, one method and one seed, with its bookkeeping:
  values      attribution per feature (+ pushes the explained output UP, i.e. towards the target class)
  fx          model output for the instance (P(target))
  base_value  the reference output the attributions are measured from (v(empty) / IG baseline output)
  n_evals     model evaluations used (cost), seconds (wall time, cost)
  extras      method-specific checks: SHAP efficiency error, IG completeness error, LIME surrogate R^2 ...
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from xaicomp.paths import safe_path


@dataclass
class Explanation:
    method: str
    instance_id: str
    seed: int
    values: np.ndarray
    fx: float
    base_value: float
    n_evals: int
    seconds: float
    target: str = "high"
    variant: str = ""                    # "" = original input; "noise0" = robustness copy
    extras: dict = field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        p = safe_path(path)
        meta = {k: v for k, v in asdict(self).items() if k != "values"}
        np.savez(p, values=np.asarray(self.values, dtype=np.float64), meta=json.dumps(meta, default=float))
        return p

    @classmethod
    def load(cls, path: Path | str) -> "Explanation":
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            return cls(values=z["values"], **meta)


def explanation_path(root: Path, instance_id: str, method: str, seed: int, variant: str = "") -> Path:
    tag = f"{method}__s{seed}" + (f"__{variant}" if variant else "")
    return root / "explanations" / instance_id / f"{tag}.npz"
