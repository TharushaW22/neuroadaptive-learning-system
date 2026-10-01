from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
from schemas import Strategy


class PolicyAlgorithm(ABC):
    name: str = "base"

    @abstractmethod
    def select(self, context: np.ndarray) -> Tuple[str, str]:
        ...

    @abstractmethod
    def update(self, context: np.ndarray, strategy: str, reward: float) -> None:
        ...

    def scores(self, context: np.ndarray) -> Dict[str, float]:
        return {s.value: 0.0 for s in Strategy}