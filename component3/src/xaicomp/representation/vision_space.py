"""Interpretable space of the vision context: 15 facial features x time segments of the 4-s window.

Component 2's GRU reads a window of T = 20 frames (4 s at 5 fps) x F = 15 per-frame features. The
window is cut into S equal time segments (default 4 segments of 5 frames = 1 s each), so one
interpretable feature is "feature f during segment s", e.g. "gaze_h|1-2 s". Removing it replaces
those L = T / S frames of that one feature.

Feature order is feature-major: index = feature_index * S + segment_index (the same layout as
EEG's channel x band, so the generic code paths work unchanged).

Group levels (summed, because attributions are additive):
  feature_x_segment  60 single features
  feature            15 (all segments of one feature)
  segment            S  (all features in one time segment)
  group              5 face regions from the context YAML (eyes/blink, gaze, head pose, mouth, brows)
  group_x_segment    5 x S
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VISION_LEVELS = ("feature_x_segment", "feature", "segment", "group", "group_x_segment")


@dataclass
class VisionSpace:
    features: list[str]
    segments: list[str]
    groups_map: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        listed = [f for fs in self.groups_map.values() for f in fs]
        unknown = [f for f in listed if f not in self.features]
        if unknown:
            raise ValueError(f"group features not in the feature list: {unknown}")
        dup = {f for f in listed if listed.count(f) > 1}
        if dup:
            raise ValueError(f"features in more than one group: {sorted(dup)}")

    @property
    def n_features(self) -> int:
        return len(self.features) * len(self.segments)

    @property
    def feature_names(self) -> list[str]:
        return [f"{f}|{s}" for f in self.features for s in self.segments]

    def index(self, feature: str, segment: str) -> int:
        return self.features.index(feature) * len(self.segments) + self.segments.index(segment)

    def groups(self, level: str) -> dict[str, list[int]]:
        """{group name: [feature indices]} for one level."""
        ns = len(self.segments)
        if level == "feature_x_segment":
            return {n: [i] for i, n in enumerate(self.feature_names)}
        if level == "feature":
            return {f: [fi * ns + s for s in range(ns)] for fi, f in enumerate(self.features)}
        if level == "segment":
            return {s: [fi * ns + si for fi in range(len(self.features))] for si, s in enumerate(self.segments)}
        if level == "group":
            return {g: [self.index(f, s) for f in fs for s in self.segments] for g, fs in self.groups_map.items()}
        if level == "group_x_segment":
            return {f"{g}|{s}": [self.index(f, s) for f in fs] for g, fs in self.groups_map.items() for s in self.segments}
        raise ValueError(f"unknown level {level!r}; choose from {VISION_LEVELS}")

    def aggregate(self, values: np.ndarray, level: str) -> tuple[np.ndarray, list[str]]:
        """Sum attributions (..., n_features) within each group -> (..., n_groups), names."""
        g = self.groups(level)
        v = np.asarray(values)
        return np.stack([v[..., idx].sum(axis=-1) for idx in g.values()], axis=-1), list(g)

    def as_matrix(self, values: np.ndarray) -> np.ndarray:
        """(..., 60) -> (..., n_features, n_segments)."""
        return np.asarray(values).reshape(*np.shape(values)[:-1], len(self.features), len(self.segments))

    def describe(self) -> dict:
        """Fields written to run_info.json (the dashboard and the C4 export rebuild the space from them)."""
        return {"space_kind": "vision", "feature_names": self.feature_names, "features": self.features,
                "segments": self.segments, "groups": self.groups_map}


def segment_names(n_segments: int, window_frames: int, fps: float) -> list[str]:
    """['0-1 s', '1-2 s', ...] for equal segments of the window."""
    step = window_frames / n_segments / fps
    return [f"{i * step:g}-{(i + 1) * step:g} s" for i in range(n_segments)]


def space_from_info(info: dict):
    """Rebuild the interpretable space of a context from its run_info.json (EEG or vision)."""
    if info.get("space_kind") == "vision":
        return VisionSpace(info["features"], info["segments"], info["groups"])
    from xaicomp.representation.groups import InterpretableSpace
    return InterpretableSpace(info["channels"], info["bands"], info["regions"])
