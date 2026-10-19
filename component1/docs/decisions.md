# Design decisions

Judgment calls made while building the engine, with the reason for each. Numbered for reference
in the viva notes and code comments.

## D1 — Proposal document not available
`docs/proposal.docx` is not in the repository. All methods follow the phase specifications
given by the author (which summarise proposal sections 4.6–4.15). If the proposal differs on a
detail, the code is config-driven and can be adjusted.

## D2 — Python 3.13 instead of 3.11
Only Python 3.13.7 is installed on the development machine; all dependencies support it.
`requires-python >= 3.11`, so 3.11 also works.

## D3 — component1/ is the eeg-engine root
The STEW data was already placed in `component1/data/raw/STEW/`, and the git repository is
shared with teammates, so `component1/` plays the role of `eeg-engine/`.

## D4 — HF/EMG quality check uses 30–45 Hz, not > 45 Hz
The Emotiv EPOC low-passes on board at ~43 Hz, so power above 45 Hz is ~0 even in noisy data
(measured ratio ~0.001). The EMG check therefore uses P(30–45 Hz) / P(1–45 Hz).

## D5 — Montage: colin27_1020 (= standard_1020)
MNE 1.13 deprecates the name `standard_1020` in favour of `colin27_1020`. For our 14 Emotiv
electrodes the positions of `colin27_1020`, `standard_1020` and `standard_1005` are identical
(max difference 0.0 mm, checked in Phase 2). The code uses `colin27_1020` when available and
falls back to `standard_1020` on older MNE, so results do not depend on the MNE version.

## D6 — Quality checks on detrended windows
The amplitude/variance checks linearly detrend each window first. Without this, slow drift
inside a 4 s window counted as an "artifact", and the result depended on how much 0.25–0.5 Hz
drift the high-pass let through (MNE's FIR has a 0.5 Hz-wide transition band, so it passes more
drift than the 4th-order Butterworth used in Phase 1).

## D7 — Amplitude threshold 150 µV + frontal blink tolerance
Threshold sweep on the filtered data (results/preprocessing/quality_raw_vs_filtered.json):
100 µV marks ~26 % of windows bad, 150 µV ~14 %, 200 µV ~10 %. 100 µV is too strict for a
consumer headset with dry-ish saline sensors; 150 µV is the common upper bound for "clean" EEG.
Frontal channels (AF3/AF4/F7/F8) that fail only amplitude/variance below 300 µV are treated as
blinks: the window becomes "warn", never "bad", because blinks are expected and the model must
cope with them (they are also present at inference time).

## D8 — Persistent bad channels: detect + spherical-spline interpolation
Rule (bad_channels.py): flat or hard-failing in >= 60 % of windows, or a median per-window std
> 4x that of the median channel. The median (not the whole-recording std) is used so a few
huge spikes do not condemn a whole channel — those windows are flagged instead. At most 3 of
14 channels are interpolated per recording. Result: 30 channels in 26 recordings, every one
logged with its reason in results/preprocessing/interpolation_log.csv. No subject is dropped.

## D9 — Normalisation location
Epochs are saved un-normalised (`normalisation: none`). Signal-level options (epoch z-score,
subject rest baseline) are implemented for the deep models; the classical models use the
feature-level rest baseline (z-score each feature with the subject's own rest epochs), which is
the stronger and more common form. Every T1 result is reported with and without it.

## D10 — Usable epochs
"Usable" = quality flag != "bad". Overall 87.8 % usable; sub27_hi and sub37_lo are below 50 %
but are kept. Experiments train on all epochs by default (the model sees realistic noise);
`exclude_bad_windows` is an experiment option.

## D11 — FIR filter edges
MNE's default transition bands are 0.5 Hz at the 0.5 Hz edge and 11.25 Hz at the 45 Hz edge
(–6 dB at ~50.6 Hz). Combined with the 50 Hz notch, power at 57–64 Hz drops by roughly 50–60 dB
(results/preprocessing/psd_before_after_*.png).
