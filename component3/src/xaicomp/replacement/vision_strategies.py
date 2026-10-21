"""How an ABSENT vision feature ("feature f during time segment s") is filled in.

The instance is held as x (F, S, L): F = 15 standardised features, S segments, L frames per segment
(F x S x L = the GRU's 15 x 20 window, re-arranged). A mask row (F*S,) marks PRESENT features.

  zero        off-manifold: the frames are set to 0 = the TRAINING MEAN of that feature (C2 standardises
              with training-split mean/std, so 0 is the average training frame). Also IG's zero baseline.
  donor       on-manifold: the frames are copied from a DONOR window of a training-split subject
              (KernelSHAP, LIME, IG donor baseline); the output is averaged over K donors.
  conditional Gaussian conditional (D2 analogue): z = the 60 segment means is modelled as N(mu, Sigma)
              with a Ledoit-Wolf covariance fitted on training-subject windows. Absent segment means are
              drawn from z_A | z_S = (instance's present means); the donor's frames are SHIFTED so their
              segment mean equals the draw (the donor keeps its within-segment time course).
  empirical   empirical conditional: the K training windows nearest to the instance on the PRESENT
              segment means are used as donors, unshifted (fallback when the Gaussian fits poorly).

`gaussian_suitability` decides between conditional and empirical from the data (configurable rule,
recorded in run_info.json), because some features are not Gaussian: blink_event is 0/1 per frame,
and quality gating fills gaps with training medians.
"""

from __future__ import annotations

import numpy as np

from xaicomp.replacement.strategies import GaussianLogPower

# The Gaussian class in strategies.py is a plain N(mu, Sigma) with Ledoit-Wolf shrinkage and standard
# conditioning; it is reused here on segment means (its name refers to its first use on EEG log power).
GaussianVector = GaussianLogPower


def segment_means(x: np.ndarray) -> np.ndarray:
    """(..., F, S, L) -> (..., F*S) mean of each segment's frames."""
    m = np.asarray(x).mean(axis=-1)
    return m.reshape(*m.shape[:-2], -1)


def compose_frames(x: np.ndarray, masks: np.ndarray, repl: np.ndarray | None, shift: np.ndarray | None = None) -> np.ndarray:
    """Windows (n, F, S, L) from instance x (F, S, L) and masks (n, F*S) of present features.

    repl: (F, S, L) donor frames (None = zero = training mean); shift: optional (n, F, S) added to repl."""
    F, S, L = x.shape
    m = np.asarray(masks, dtype=np.float32).reshape(len(masks), F, S, 1)
    out = m * x[None]
    if repl is not None:
        r = repl[None] if shift is None else repl[None] + shift[..., None]
        out = out + (1.0 - m) * r
    return out.astype(np.float32)


def conditional_shifts(masks: np.ndarray, z_x: np.ndarray, donor_z: np.ndarray, gauss: GaussianVector,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """For every mask row and donor k: additive shifts (n, K, F*S) that move donor k's absent segments to a
    conditional draw of their mean, plus the sampled targets (n, K, F*S) (NaN where present)."""
    n, D = masks.shape
    K = len(donor_z)
    shifts = np.zeros((n, K, D))
    targets = np.full((n, K, D), np.nan)
    for i in range(n):
        present = masks[i].astype(bool)
        if present.all():
            continue
        draws = gauss.sample_absent(present, z_x, K, rng)          # (K, n_absent)
        A = ~present
        targets[i][:, A] = draws
        shifts[i][:, A] = draws - donor_z[:, A]
    return shifts.astype(np.float32), targets


def gaussian_suitability(Z: np.ndarray, groups: np.ndarray | None = None, n_masks: int = 20,
                         nominal: float = 0.90, coverage_range=(0.85, 0.95), max_discrete_share: float = 0.10,
                         seed: int = 0) -> dict:
    """Is a Gaussian a fair model of the segment-mean vector z? (decides Conditional SHAP's mode)

    1. Calibration on held-out windows: fit on 80 % of the training windows (split by subject when
       `groups` is given), then for random masks check how often a held-out window's TRUE absent value lies
       inside the central `nominal` interval of z_A | z_S. A well-specified Gaussian gives ~`nominal`.
    2. Discreteness: share of dimensions with <= 6 distinct values in training (e.g. blink_event).
    Gaussian is used when coverage is inside `coverage_range` and the discrete share <= max_discrete_share;
    otherwise the empirical nearest-donor conditional is used. Every number is returned for the record."""
    from scipy.stats import norm

    rng = np.random.default_rng([seed, 77])
    Z = np.asarray(Z, dtype=np.float64)
    if groups is not None:
        ug = np.unique(groups)
        hold_g = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False).tolist())
        hold = np.array([g in hold_g for g in groups])
    else:
        hold = rng.random(len(Z)) < 0.2
    if hold.all() or (~hold).sum() < Z.shape[1] + 2 or hold.sum() == 0:
        return {"usable": False, "mode": "empirical", "reason": "too few windows to check the Gaussian fit",
                "n_windows": int(len(Z))}
    g = GaussianVector(Z[~hold])
    zcrit = float(norm.ppf(0.5 + nominal / 2))
    inside, total = 0, 0
    H = Z[hold][: 200]
    for _ in range(n_masks):
        present = rng.random(Z.shape[1]) < 0.5
        if present.all() or not present.any():
            continue
        for z in H:
            mean, cov = g.conditional(present, z[present])
            sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            inside += int((np.abs(z[~present] - mean) <= zcrit * sd).sum())
            total += int((~present).sum())
    coverage = inside / total if total else float("nan")
    n_unique = np.array([len(np.unique(np.round(c, 6))) for c in Z[~hold].T])
    discrete = float((n_unique <= 6).mean())
    ok = bool(coverage == coverage and coverage_range[0] <= coverage <= coverage_range[1] and discrete <= max_discrete_share)
    return {"usable": ok, "mode": "gaussian" if ok else "empirical", "coverage": float(coverage), "nominal": nominal,
            "coverage_range": list(coverage_range), "discrete_share": discrete, "max_discrete_share": max_discrete_share,
            "discrete_dims": np.where(n_unique <= 6)[0].tolist(), "shrinkage": g.shrinkage,
            "n_fit": int((~hold).sum()), "n_holdout_used": int(len(H)),
            "reason": ("Gaussian calibrated on held-out training windows" if ok else
                       "Gaussian poorly calibrated or too many discrete features -> empirical nearest-donor conditional")}
