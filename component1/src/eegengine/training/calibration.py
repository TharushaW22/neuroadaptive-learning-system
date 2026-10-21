"""Temperature scaling (Guo et al., 2017): divide logits by one scalar T > 0.

T is fit on the inner VALIDATION split (subjects not used for the gradient updates and never the
test subjects) by minimising the negative log-likelihood. T > 1 softens over-confident
predictions; argmax — and therefore accuracy/F1 — is unchanged, only the probabilities move.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    lg = torch.as_tensor(logits, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)     # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
