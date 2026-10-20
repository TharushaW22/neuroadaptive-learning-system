"""How an ABSENT interpretable feature (one channel x band component) is filled in.

  zero        off-manifold: the component is set to 0 (the band is simply missing).
  donor       marginal / interventional: the component is taken from a DONOR window of another
              subject in the instance's fold-training set; the model output is averaged over K donors.
              Used by KernelSHAP, LIME and IG (donor baseline).
  conditional Gaussian conditional (D2): the 70-d vector z of log10 band powers is modelled as
              N(mu, Sigma) with a Ledoit-Wolf covariance fitted on the fold's TRAINING subjects only.
              Absent powers are sampled from z_A | z_S = (instance's present powers); the donor
              waveform is then RESCALED so its power equals the sampled target. Used by
              Conditional SHAP. Limitation: only power is conditioned; phase and cross-channel
              connectivity of the donor waveform are not.
  empirical   empirical conditional (D2 alternative): the K donors whose band-power profile is
              nearest on the PRESENT features (standardised Euclidean) are used without rescaling.

All functions build model inputs as  segment = residual + sum_b [mask*x_b + (1-mask)*r_b]
with arrays shaped (channels, bands, time); masks are (n, channels*bands) channel-major.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def masks_to_cb(masks: np.ndarray, n_ch: int, n_b: int) -> np.ndarray:
    return np.asarray(masks, dtype=np.float32).reshape(len(masks), n_ch, n_b)


def compose(x: np.ndarray, residual: np.ndarray, masks: np.ndarray, repl: np.ndarray | None,
            repl_scale: np.ndarray | None = None) -> np.ndarray:
    """Segments (n, C, T) from instance components x (C, B, T), residual (C, T), masks (n, C*B).

    repl: (C, B, T) replacement components (None = zero); repl_scale: optional (n, C, B) gain on repl.
    """
    C, B, T = x.shape
    m = masks_to_cb(masks, C, B)                                       # (n, C, B)
    seg = np.einsum("ncb,cbt->nct", m, x, optimize=True) + residual[None]
    if repl is not None:
        w = (1.0 - m) if repl_scale is None else (1.0 - m) * repl_scale
        seg += np.einsum("ncb,cbt->nct", w.astype(np.float32), repl, optimize=True)
    return seg.astype(np.float32)


class GaussianLogPower:
    """N(mu, Sigma) over the 70 log band powers, Ledoit-Wolf shrinkage, fitted on training windows."""

    def __init__(self, Z_train: np.ndarray):
        lw = LedoitWolf().fit(Z_train)
        self.mu = lw.location_.astype(np.float64)
        self.cov = lw.covariance_.astype(np.float64)
        self.shrinkage = float(lw.shrinkage_)
        self.n_train = len(Z_train)

    def conditional(self, present: np.ndarray, z_present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of z_absent | z_present (standard Gaussian conditioning)."""
        A, S = ~present, present
        if not S.any():
            return self.mu[A].copy(), self.cov[np.ix_(A, A)].copy()
        Sss = self.cov[np.ix_(S, S)]
        Sas = self.cov[np.ix_(A, S)]
        K = np.linalg.solve(Sss, Sas.T).T                               # Sigma_AS Sigma_SS^-1
        mean = self.mu[A] + K @ (z_present - self.mu[S])
        cov = self.cov[np.ix_(A, A)] - K @ Sas.T
        return mean, 0.5 * (cov + cov.T)

    def sample_absent(self, present: np.ndarray, z_x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """n draws of the absent log powers given the instance's present ones -> (n, n_absent)."""
        mean, cov = self.conditional(present, z_x[present])
        if mean.size == 0:
            return np.zeros((n, 0))
        jitter = 1e-9 * np.eye(len(mean))
        try:
            L = np.linalg.cholesky(cov + jitter)
        except np.linalg.LinAlgError:                                    # numerically not PD: eigen fix
            w, V = np.linalg.eigh(cov)
            L = V * np.sqrt(np.clip(w, 0, None))
        return mean + rng.standard_normal((n, len(mean))) @ L.T


def conditional_gains(masks: np.ndarray, z_x: np.ndarray, donor_z: np.ndarray, gauss: GaussianLogPower,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """For every mask row and donor k: amplitude gains (n, K, F) that rescale donor k's absent
    components to a conditional draw of their log power, plus the sampled targets (n, K, F) (NaN where present)."""
    n, F = masks.shape
    K = len(donor_z)
    gains = np.ones((n, K, F), dtype=np.float64)
    targets = np.full((n, K, F), np.nan)
    for i in range(n):
        present = masks[i].astype(bool)
        if present.all():
            continue
        draws = gauss.sample_absent(present, z_x, K, rng)                # (K, n_absent)
        A = ~present
        targets[i][:, A] = draws
        gains[i][:, A] = np.sqrt(10.0 ** (draws - donor_z[:, A]))         # power ratio -> amplitude gain
    return gains.astype(np.float32), targets


def nearest_donors(present: np.ndarray, z_x: np.ndarray, pool_z: np.ndarray, scale: np.ndarray, k: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Indices of the k pool windows closest to the instance on the PRESENT features (empirical conditional)."""
    if not present.any():
        return rng.choice(len(pool_z), size=k, replace=False)
    d = (((pool_z[:, present] - z_x[present]) / scale[present]) ** 2).sum(axis=1)
    return np.argsort(d, kind="stable")[:k]
