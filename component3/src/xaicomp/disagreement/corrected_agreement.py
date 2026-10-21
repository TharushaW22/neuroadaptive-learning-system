"""Noise-corrected agreement between methods (classical correction for attenuation, Spearman 1904).

Two noisy methods can never agree more than their own run-to-run reliability allows. With
rho_AB = agreement between methods A and B (same seed), and rho_AA, rho_BB = each method's
agreement with itself across seeds (stability):

    rho_AB_corrected = rho_AB / sqrt(rho_AA * rho_BB)

The corrected value estimates how much the methods would agree if both were noise-free.
Values > 1 are possible when the self-agreements are small/noisy; they are FLAGGED, not clipped.
IG with a zero baseline is deterministic (rho_AA = 1 trivially), so it would inflate the corrected
value: pairs with ig_zero are reported raw only (D3); the donor-baseline IG is used for correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DETERMINISTIC = {"ig_zero"}


def corrected_agreement(agreement: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """Per method pair: raw mean agreement, mean self-agreements, corrected agreement (ratio of means)."""
    ag = agreement.groupby(["instance_id", "method_a", "method_b"])["rank_corr"].mean().reset_index()
    st = stability.set_index(["instance_id", "method"])["stability"].sort_index()
    rows = []
    for (a, b), g in ag.groupby(["method_a", "method_b"]):
        raw = float(g["rank_corr"].mean())
        saa = float(np.nanmean([st.get((i, a), np.nan) for i in g["instance_id"]]))
        sbb = float(np.nanmean([st.get((i, b), np.nan) for i in g["instance_id"]]))
        det = a in DETERMINISTIC or b in DETERMINISTIC
        corr = raw / np.sqrt(saa * sbb) if (not det and saa > 0 and sbb > 0) else np.nan
        rows.append({"method_a": a, "method_b": b, "n_instances": len(g), "raw_agreement": raw,
                     "self_agreement_a": saa, "self_agreement_b": sbb, "corrected_agreement": corr,
                     "corrected_gt_1": bool(corr == corr and corr > 1),
                     "note": "raw only: deterministic method (D3)" if det else
                     ("not correctable: self-agreement <= 0" if corr != corr else "")})
    return pd.DataFrame(rows)
