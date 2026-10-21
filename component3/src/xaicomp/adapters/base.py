"""The ModelAdapter interface: the ONLY thing explainers, metrics and statistics see of a model.

An adapter turns one modality + model into:
  * an interpretable space (named features + groupings),
  * a set of real instances with subject IDs, folds and labels,
  * an `InstanceView` per instance: evaluate the model on any coalition (mask over the
    interpretable features) with a chosen replacement strategy, and optionally a differentiable
    torch module for gradient methods.

EEG (Component 1) is implemented in eeg_component1.py. Vision and multimodal adapters for
Component 2 implement the same interface later (D8): explainers, metrics, statistics, ranking,
export and dashboard need no changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xaicomp.representation.groups import InterpretableSpace


@dataclass
class Capabilities:
    differentiable: bool                 # a torch module from interpretable components to the output exists
    n_features: int
    model_kind: str                      # "deep" | "tree" | ...
    has_subject_ids: bool = True
    supports_conditional: bool = True    # a conditional replacement distribution can be fitted
    output: str = "probability"
    est_ms_per_eval: float | None = None
    notes: dict = field(default_factory=dict)


class InstanceView(ABC):
    """One explained instance. Everything an explainer may do with the model goes through here."""

    instance_id: str
    fold: int
    n_features: int

    @abstractmethod
    def value(self, masks: np.ndarray, replacement: str, rng: np.random.Generator, **kw) -> np.ndarray:
        """Model output (target-class probability) for coalitions `masks` (n, n_features) of
        PRESENT features; absent ones are filled by `replacement` (zero | donor | conditional | empirical).
        Returns (n,), averaged over the replacement draws."""

    @abstractmethod
    def fx(self) -> float:
        """Model output on the unmodified instance."""

    def torch_parts(self):
        """(module, inputs, extra args) for gradient methods; adapters without a differentiable
        path keep this default, and the screening marks gradient methods 'not applicable'."""
        raise NotImplementedError("model is not differentiable through this adapter")


class ModelAdapter(ABC):
    context_id: str
    modality: str
    class_names: list[str]
    target_class: str
    space: InterpretableSpace

    @abstractmethod
    def instances(self) -> pd.DataFrame:
        """All candidate instances: instance_id, subject_id, fold, y, condition, frac_bad, ..."""

    @abstractmethod
    def predict_instances(self, instance_ids: list[str]) -> np.ndarray:
        """Calibrated class probabilities (n, n_classes) for the unmodified instances."""

    @abstractmethod
    def view(self, instance_id: str, k_donors: int, seed: int) -> InstanceView:
        """Prepare one instance for explanation (donors drawn from the instance's fold training subjects)."""

    @abstractmethod
    def capabilities(self) -> Capabilities:
        ...

    @property
    @abstractmethod
    def in_sample(self) -> bool:
        """True when the explained model has seen the instance's subject in training."""
