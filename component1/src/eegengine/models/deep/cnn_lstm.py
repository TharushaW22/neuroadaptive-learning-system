"""CNN-LSTM baseline: per-epoch EEGNet-style CNN encoder -> LSTM over the epoch sequence.

It is exactly the hybrid model with only the CNN and LSTM branches switched on (ablation A7),
so the baseline and the ablation share one implementation and cannot drift apart.
"""

from __future__ import annotations

from eegengine.models.deep.hybrid import HybridModel


def build_cnn_lstm(n_classes: int, **kwargs) -> HybridModel:
    return HybridModel(n_classes, branches=("cnn", "lstm"), **kwargs)
