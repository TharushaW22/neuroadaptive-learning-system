"""The interpretable feature space and its groupings (channel, band, region, channel x band).

Feature order is channel-major: index f = channel_index * n_bands + band_index, names "O1|alpha".
Attributions are additive (SHAP efficiency, IG completeness), so a group's attribution is the
SUM of its features' attributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEVELS = ("channel_x_band", "channel", "band", "region", "region_x_band")


@dataclass
class InterpretableSpace:
    channels: list[str]
    bands: list[str]
    regions: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        unknown = [c for chs in self.regions.values() for c in chs if c not in self.channels]
        if unknown:
            raise ValueError(f"region channels not in montage: {unknown}")

    @property
    def n_features(self) -> int:
        return len(self.channels) * len(self.bands)

    @property
    def feature_names(self) -> list[str]:
        return [f"{c}|{b}" for c in self.channels for b in self.bands]

    def index(self, channel: str, band: str) -> int:
        return self.channels.index(channel) * len(self.bands) + self.bands.index(band)

    def groups(self, level: str) -> dict[str, list[int]]:
        """{group name: [feature indices]} for one level."""
        nb = len(self.bands)
        if level == "channel_x_band":
            return {n: [i] for i, n in enumerate(self.feature_names)}
        if level == "channel":
            return {c: [ci * nb + b for b in range(nb)] for ci, c in enumerate(self.channels)}
        if level == "band":
            return {b: [ci * nb + bi for ci in range(len(self.channels))] for bi, b in enumerate(self.bands)}
        if level == "region":
            return {r: [self.index(c, b) for c in chs for b in self.bands] for r, chs in self.regions.items()}
        if level == "region_x_band":
            return {f"{r}|{b}": [self.index(c, b) for c in chs] for r, chs in self.regions.items() for b in self.bands}
        raise ValueError(f"unknown level {level!r}; choose from {LEVELS}")

    def aggregate(self, values: np.ndarray, level: str) -> tuple[np.ndarray, list[str]]:
        """Sum attributions (..., n_features) within each group -> (..., n_groups), names."""
        g = self.groups(level)
        v = np.asarray(values)
        return np.stack([v[..., idx].sum(axis=-1) for idx in g.values()], axis=-1), list(g)

    def as_matrix(self, values: np.ndarray) -> np.ndarray:
        """(..., 70) -> (..., n_channels, n_bands)."""
        return np.asarray(values).reshape(*np.shape(values)[:-1], len(self.channels), len(self.bands))
