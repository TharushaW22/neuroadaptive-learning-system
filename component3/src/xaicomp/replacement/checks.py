"""D2 check: does the composed signal really have the band power Conditional SHAP sampled?

Conditional SHAP rescales each absent donor component so its log10 power equals a draw from
z_absent | z_present. The model, however, sees the SUMMED signal. This check re-decomposes the
composed window with the same filter bank and compares the re-measured log power of every absent
component with its sampled target. The same re-measurement applied to the UNMODIFIED window gives
the error that comes from re-decomposing an 18-s window at all (edge effects, crossover leakage);
the excess over that baseline is the error caused by the rescaling itself.
"""

from __future__ import annotations

import numpy as np

from xaicomp.replacement.strategies import compose, conditional_gains
from xaicomp.representation.eeg_bands import decompose, log_power


def conditional_power_check(view, n_masks: int = 8, p_present: float = 0.5, seed: int = 0, pad: int = 512) -> dict:
    ad = view.ad
    rng = np.random.default_rng([seed, 31])
    masks = rng.random((n_masks, view.n_features)) < p_present
    gains, targets = conditional_gains(masks, view.z_x, view.donor_z, ad.gaussian(view.fold), rng)
    errs = []
    for i in range(n_masks):
        for k in range(view.k_donors):
            seg = compose(view.x, view.residual, masks[i:i + 1], view.donors[k],
                          gains[i:i + 1, k].reshape(1, view.C, view.B))[0]
            comps, _ = decompose(seg, ad.data.fs, ad.bands, ad.crossover, pad=pad)
            z = log_power(comps).reshape(-1)
            A = ~masks[i]
            errs.append(z[A] - targets[i, k, A])
    errs = np.concatenate(errs)
    comps0, _ = decompose(view.segment(), ad.data.fs, ad.bands, ad.crossover, pad=pad)
    base = log_power(comps0).reshape(-1) - view.z_x
    per_band = {}
    ab = np.abs(errs)
    return {
        "n_components_checked": int(errs.size),
        "mean_abs_err_log10": float(ab.mean()), "median_abs_err_log10": float(np.median(ab)),
        "p95_abs_err_log10": float(np.percentile(ab, 95)), "mean_abs_err_dB": float(10 * ab.mean()),
        "baseline_mean_abs_err_log10": float(np.abs(base).mean()),
        "baseline_p95_abs_err_log10": float(np.percentile(np.abs(base), 95)),
        "bias_log10": float(errs.mean()),
        **per_band,
    }
