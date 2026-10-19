"""Optional ICA ocular-artifact removal (default OFF in preprocess.yaml).

The Emotiv EPOC has no EOG electrodes, so the most frontal channels AF3/AF4 act as EOG proxies:
an independent component whose time course correlates strongly with them is treated as ocular
and removed. FastICA is used (bundled with scikit-learn, so no extra install).
"""

from __future__ import annotations

import numpy as np


def run_ica(data: np.ndarray, fs: float, ch_names: list[str], cfg: dict, seed: int = 42):
    """Return (cleaned_data, log). `data` is filtered EEG in microvolts."""
    import mne

    from eegengine.io.stew import _montage_name

    info = mne.create_info(list(ch_names), sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage(_montage_name()))

    # ICA is fit on a 1 Hz high-passed copy (slow drifts hurt the decomposition), then applied
    # to the original 0.5 Hz data.
    fit_raw = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    n_comp = cfg.get("n_components", len(ch_names) - 1)
    ica = mne.preprocessing.ICA(n_components=n_comp, method=cfg.get("method", "fastica"),
                                random_state=seed, max_iter="auto", verbose="ERROR")
    ica.fit(fit_raw, verbose="ERROR")
    eog_chs = cfg.get("eog_proxies", ["AF3", "AF4"])
    idx, scores = ica.find_bads_eog(raw, ch_name=eog_chs, threshold=cfg.get("eog_z_thresh", 3.0),
                                    verbose="ERROR")
    idx = idx[: cfg.get("max_removed", 2)]
    ica.exclude = list(idx)
    cleaned = ica.apply(raw.copy(), verbose="ERROR").get_data() * 1e6
    scores = np.atleast_2d(scores)
    log = {
        "n_components": int(ica.n_components_),
        "removed_components": [int(i) for i in idx],
        "eog_scores_max_abs": [round(float(np.abs(scores[:, i]).max()), 3) for i in idx],
        "eog_proxies": eog_chs,
    }
    return cleaned, log
