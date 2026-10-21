"""Label schemes. The original DAiSEE label is always kept in the feature table.

* daisee_original  - 4 classes: the target's intensity level 0..3 (default experiment)
* daisee_binary    - low (< threshold) vs high (>= threshold) intensity
* operational      - FOCUSED / REDUCED_ATTENTION / DISTRACTED. Disabled unless an explicit
                     mapping AND an existing written justification document are configured.
                     Low engagement is NOT automatically treated as distraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.data.feature_schema import LABEL_TARGETS

OPERATIONAL_CATEGORIES = ("FOCUSED", "REDUCED_ATTENTION", "DISTRACTED")


class LabelSchemeError(ValueError):
    pass


@dataclass
class LabelScheme:
    name: str
    target: str
    column: str
    class_names: list[str]
    mapping: dict[int, int] = field(default_factory=dict)   # original level -> class index
    justification_document: str | None = None

    def transform(self, level) -> int:
        lvl = int(level)
        if lvl not in self.mapping:
            raise LabelSchemeError(f"Level {lvl} not covered by scheme {self.name}")
        return self.mapping[lvl]

    def to_dict(self) -> dict:
        return {"name": self.name, "target": self.target, "column": self.column,
                "class_names": self.class_names, "mapping": {str(k): v for k, v in self.mapping.items()},
                "justification_document": self.justification_document}


def get_label_scheme(cfg: dict) -> LabelScheme:
    lc = cfg["labels"]
    target = lc.get("target", "engagement")
    if target not in LABEL_TARGETS:
        raise LabelSchemeError(f"Unknown label target '{target}'")
    col = LABEL_TARGETS[target]
    scheme = lc.get("scheme", "daisee_original")
    if scheme == "daisee_original":
        return LabelScheme(scheme, target, col, [f"{target}_{i}" for i in range(4)], {i: i for i in range(4)})
    if scheme == "daisee_binary":
        t = int(lc.get("binary_threshold", 2))
        if not 1 <= t <= 3:
            raise LabelSchemeError("binary_threshold must be 1, 2 or 3")
        return LabelScheme(scheme, target, col, [f"{target}_low(<{t})", f"{target}_high(>={t})"],
                           {i: int(i >= t) for i in range(4)})
    if scheme == "operational":
        mapping = lc.get("operational_mapping")
        doc = lc.get("operational_mapping_document")
        if not mapping or not doc:
            raise LabelSchemeError(
                "Operational categories require labels.operational_mapping AND "
                "labels.operational_mapping_document (a written, defensible protocol). "
                "See docs/labeling_protocol.md. Refusing to equate DAiSEE levels with distraction by default.")
        doc_path = Path(doc)
        if not doc_path.is_absolute():
            doc_path = Path(cfg.get("_project_root", ".")) / doc_path
        if not doc_path.exists():
            raise LabelSchemeError(f"Justification document not found: {doc_path}")
        names = [c for c in OPERATIONAL_CATEGORIES if c in set(mapping.values())]
        bad = set(mapping.values()) - set(OPERATIONAL_CATEGORIES)
        if bad:
            raise LabelSchemeError(f"Unknown operational categories: {bad}")
        return LabelScheme(scheme, target, col, names,
                           {int(k): names.index(v) for k, v in mapping.items()}, str(doc_path))
    raise LabelSchemeError(f"Unknown label scheme '{scheme}'")
