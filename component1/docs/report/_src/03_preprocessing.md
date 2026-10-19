# 3. Preprocessing and signal quality

All steps are driven by `configs/preprocess.yaml` (reproduced in full in the appendix) and run by
`src/eegengine/preprocess/pipeline.py:preprocess_recording` for each recording, from
`scripts/run_preprocess.py`. Order:

```
raw (µV, DC offset) → mean removal → FIR band-pass 0.5–45 Hz → FIR notch 50 Hz
  → persistent bad-channel detection + spherical-spline interpolation → (optional ICA, OFF)
  → per-window quality flags → 4-s epochs, 2-s step (+ metadata) → normalisation option
```
Output: `data/processed/epochs_6d7dd612.npz` with `X` of shape 7104 × 14 × 512
(`results/preprocessing/preprocess_summary.json` → `X_shape`), plus a metadata table (one row per epoch:
subject, condition, rating, epoch index, t_start, t_end, quality flag, quality score, bad channels,
interpolated channels, y_binary, y_level, y_within). The file name contains a SHA-1 hash of the
configuration (`pipeline.py:config_hash`), so a changed setting can never silently reuse old epochs.

## 3.1 DC removal
* **What:** subtract each channel's mean over the recording, `x'(t) = x(t) − mean(x)`.
* **Why:** the Emotiv adds a constant offset of about 4333 µV (mean 4333.0 [`results/eda/dataset_summary.json` → `raw_dc_offset_uv.mean`])
  that has nothing to do with brain activity and would dominate every amplitude-based computation.
* **How:** `src/eegengine/preprocess/filters.py:filter_recording` (first line). For the raw quality report
  and EDA, `src/eegengine/preprocess/dc.py:remove_dc` additionally applies a 4th-order zero-phase
  Butterworth high-pass at 0.5 Hz (`scipy.signal.sosfiltfilt`).
* **Limitation:** mean removal alone leaves slow drift; that is removed by the band-pass below.

## 3.2 Band-pass 0.5–45 Hz (zero-phase FIR)
* **What:** a linear-phase finite-impulse-response filter that keeps 0.5–45 Hz. `mne.filter.filter_data`
  designs a windowed-sinc filter (`fir_design="firwin"`, Hamming window) and applies it with
  `phase="zero"`, i.e. the filter delay is compensated so peaks are not shifted in time.
* **Parameters:** l_freq 0.5 [`configs/preprocess.yaml` → `filter.l_freq`] Hz, h_freq 45.0 [`configs/preprocess.yaml` → `filter.h_freq`] Hz,
  padding `reflect_limited`. With MNE's automatic rules the transition bands are 0.5 Hz at the low edge and
  11.25 Hz at the high edge (−6 dB at ≈ 50.6 Hz) and the filter length is chosen automatically by MNE
  (≈ 3.3 / 0.5 Hz ≈ 6.6 s for a Hamming window) — derived from MNE's documented defaults, `docs/decisions.md` D11.
* **Why these edges:** 0.5 Hz removes electrode drift and sweat potentials but keeps delta; 45 Hz keeps
  the gamma band used by the features (30–45 Hz) while staying below the 50 Hz mains frequency and the
  Emotiv's own ~43 Hz low-pass. **Why FIR + zero phase:** linear phase does not distort the waveform shape
  or the phase relations between channels — essential because PLV (§4.4) measures phase differences. An
  IIR (Butterworth) filter would be shorter but has non-linear phase unless run forwards-backwards.
* **How:** `src/eegengine/preprocess/filters.py:bandpass`. Input/output `(channels, samples)` µV.
* **Produced:** see the before/after PSDs in §3.9. Tested in `tests/test_preprocess.py`
  (`test_bandpass_passes_in_band`: gain 0.97–1.03 at 10/30 Hz; `test_bandpass_attenuates_out_of_band`:
  gain < 0.1 at 0.05 and 58 Hz; `test_bandpass_is_zero_phase`: lag 0 samples).
* **Limitation:** the long (several-second) filter needs context; in real time the engine filters the
  last 20 s and keeps the newest 4 s (§8.2). Band powers agree with offline filtering within 5 %
  (`tests/test_engine_api.py:test_engine_band_power_close_to_offline`).

## 3.3 Notch 50 Hz
* **What:** a narrow FIR band-stop at the mains frequency (`mne.filter.notch_filter`, zero phase).
* **Why:** Sri Lanka's mains electricity is 50 Hz [`configs/preprocess.yaml` → `filter.notch_freq`]; power-line
  interference appears as a sharp peak there. With the 45 Hz low-pass and the Emotiv's on-board notch it is
  largely redundant for STEW, but it is cheap and protects other headsets/rooms (`configs/preprocess.yaml` comment).
* **How:** `src/eegengine/preprocess/filters.py:notch`; skipped automatically if the frequency is above Nyquist (64 Hz).
* **Tested:** `test_notch_removes_50hz` (gain < 0.1 at 50 Hz, > 0.97 at 20 Hz).

## 3.4 Signal-quality checks (per window, per channel)
* **What:** five checks on every 4-s window (2-s step) and every channel, implemented in
  `src/eegengine/quality/checks.py:channel_metrics` and `assess_window`. Each window is first linearly
  detrended (`scipy.signal.detrend`) so slow drift is not mistaken for an artefact (`docs/decisions.md` D6).

| Check | Metric | Threshold | Rationale |
|---|---|---|---|
| flat | channel std | < 0.5 [`configs/preprocess.yaml` → `quality.flat_std_uv`] µV | disconnected/saturated electrode (Emotiv LSB ≈ 0.51 µV) |
| amplitude | max abs(x) of the detrended window | > 150.0 [`configs/preprocess.yaml` → `quality.max_abs_uv`] µV | blinks, movement, electrode pops; 150 µV is a common upper bound for clean scalp EEG |
| variance | robust z-score of log-variance across the 14 channels: `z = (v − median) / (1.4826 · MAD)` | abs(z) > 3.5 [`configs/preprocess.yaml` → `quality.var_z_thresh`] | one channel far noisier/quieter than the others in the same window; median/MAD are robust to the outlier itself |
| hf (EMG) | P(30–45 Hz) / P(1–45 Hz) from Welch (1-s segments) | > 0.25 [`configs/preprocess.yaml` → `quality.hf_ratio_thresh`] | muscle (EMG) contamination is broadband high-frequency power |
| line | mean PSD within ±1 Hz of 50 Hz / mean PSD at 2–5 Hz from it | > 10.0 [`configs/preprocess.yaml` → `quality.line_ratio_thresh`] | power-line interference peak |

* **Blink tolerance** (`docs/decisions.md` D7): a failure on a frontal channel (AF3, AF4, F7, F8) that is
  *only* amplitude and/or variance with max |x| < 300.0 [`configs/preprocess.yaml` → `quality.blink_max_uv`] µV is
  counted as a blink. Blinks lower the score and make the window "warn" but never "bad": they are
  normal, frequent, and present at inference time too.
* **Window verdict:** `score = 1 − n_bad_channels / 14`; `n_hard` = bad channels that are not blink-like;
  flag = **good** (no bad channel), **bad** (n_hard > 2), otherwise **warn**. Windows are only **marked**;
  nothing is deleted. Experiments can exclude bad windows from *training* (`exclude_bad_windows`), but
  this was not used (all reported runs train on all windows).

### DEVIATION FROM PROPOSAL — EMG band 30–45 Hz instead of > 45 Hz
The first implementation measured high-frequency power above 45 Hz. On STEW that ratio was ≈ 0.001 even in
noisy recordings, because the **Emotiv EPOC low-pass filters on board at about 43 Hz**, so almost no power
exists above 45 Hz. The check was moved inside the passband (30–45 Hz) where muscle activity is visible
(`docs/decisions.md` D4).

### Quality on the raw signal (current rules)
<!-- TABLE:phase1_quality -->

### Raw vs filtered vs final flag rates
<!-- TABLE:quality_rates -->

Filtering *increases* the share of flagged windows before interpolation (good 0.355 [`results/preprocessing/quality_raw_vs_filtered.json` → `raw_dc_removed.good`]
→ 0.297 [`results/preprocessing/quality_raw_vs_filtered.json` → `filtered.good`]): the FIR's gentle 0.5 Hz transition
band passes slightly more 0.25–0.5 Hz drift than the Butterworth used for the raw report, and persistently
broken channels are still present. After interpolation of persistently bad channels the final rates are
good 0.372 [`results/preprocessing/quality_raw_vs_filtered.json` → `filtered_interpolated_final.good`],
warn 0.505 [`results/preprocessing/quality_raw_vs_filtered.json` → `filtered_interpolated_final.warn`],
bad 0.123 [`results/preprocessing/quality_raw_vs_filtered.json` → `filtered_interpolated_final.bad`].

### Choosing the amplitude threshold
<!-- TABLE:sweep -->

100 µV would mark 0.264 [`results/preprocessing/quality_raw_vs_filtered.json` → `amplitude_threshold_sweep_filtered.100.bad`]
of windows bad — too strict for a consumer headset; 200 µV only 0.095 [`results/preprocessing/quality_raw_vs_filtered.json` → `amplitude_threshold_sweep_filtered.200.bad`].
150 µV was chosen as the conventional upper bound for clean EEG (D7).

## 3.5 ICA (optional, default OFF)
* **What:** Independent Component Analysis unmixes the 14 channels into statistically independent
  sources; eye-blink sources can be removed and the signal rebuilt.
* **How:** `src/eegengine/preprocess/ica.py:run_ica` — MNE `ICA` with `method="fastica"`,
  n_components 13 [`configs/preprocess.yaml` → `ica.n_components`], fit on a 1-Hz high-passed copy, then
  `find_bads_eog` with **AF3 and AF4 as EOG proxies** (the Emotiv has no EOG electrode), z-threshold
  3.0 [`configs/preprocess.yaml` → `ica.eog_z_thresh`], at most 2 components removed; every decision is logged.
* **Why it is OFF by default:** (1) without a real EOG channel, AF3/AF4 contain frontal *brain* activity
  too — removing "AF3-like" components risks deleting frontal theta, which is the most workload-related
  rhythm; (2) with only 14 channels ICA has few components and separation is imperfect; (3) blinks are
  already handled by quality flags and will also be present in live use, so models should be robust to
  them; (4) a deterministic, simpler pipeline is easier to reproduce and defend.
* **Produced (demo run, `run_preprocess.py --ica-demo`):**
<!-- TABLE:ica -->

* **Limitation:** the effect of ICA on classification was not evaluated (NOT AVAILABLE).

## 3.6 Persistent bad channels + spherical-spline interpolation
* **What:** a channel that is broken for most of a recording is rebuilt from its neighbours with
  spherical-spline interpolation (Perrin et al., 1989, as implemented in MNE `interpolate_bads`).
* **Rule** (`src/eegengine/preprocess/bad_channels.py:detect_bad_channels`, D8): a channel is persistently
  bad if it is flat or *hard*-failing (amplitude / abnormally HIGH variance / EMG / line, blink-like
  failures excluded) in ≥ 0.6 [`configs/preprocess.yaml` → `bad_channels.min_bad_fraction`] of the windows,
  **or** its *median* per-window std is > 4.0 [`configs/preprocess.yaml` → `bad_channels.max_std_ratio`] × that of the
  median channel. The median is used so a few huge spikes cannot condemn a whole channel (those windows are
  flagged instead). Low variance alone is not a reason. At most 3 [`configs/preprocess.yaml` → `bad_channels.max_interpolated`]
  of 14 channels are rebuilt per recording (worst first).
* **How:** `bad_channels.py:interpolate_channels` builds an `mne.io.RawArray` with the 10-20 montage, sets
  `info["bads"]` and calls `interpolate_bads(mode="accurate")`.
* **Produced:** 30 [`results/preprocessing/preprocess_summary.json` → `n_channels_interpolated`] channels in
  26 [`results/preprocessing/preprocess_summary.json` → `n_recordings_with_interpolation`] recordings; none detected
  but left un-interpolated. Every decision:
<!-- TABLE:interp -->

* **Tested:** `tests/test_preprocess.py:test_bad_channel_detected_and_interpolated` (a synthetic broken T7 is
  found, rebuilt from neighbours to < 30 µV std, other channels untouched) and
  `test_pipeline_one_recording` (sub01_lo T7 is interpolated).
* **Limitation:** interpolation with only 14 sparse electrodes is approximate; frontal/temporal edge
  channels have few neighbours.

## 3.7 Epoching
* **What/how:** `src/eegengine/preprocess/epoching.py:make_epochs` cuts each cleaned recording into windows
  of 4.0 [`configs/preprocess.yaml` → `epoching.window_s`] s with a 2.0 [`configs/preprocess.yaml` → `epoching.step_s`] s
  step (50 % overlap) using a strided view: 74 [`results/preprocessing/preprocess_summary.json` → `epochs_per_recording`]
  epochs per 150-s recording, 7104 epochs in total. Quality windows use the same grid, so each epoch carries
  its own flag.
* **Why 4 s:** long enough for stable spectra (Welch with 2-s segments → 0.5 Hz resolution) and for phase
  estimates down to delta (4 cycles at 1 Hz); short enough to track changes. **Why 50 % overlap:** doubles
  the number of training windows and gives the engine an output every 2 s.
* **2-s variant:** `configs/preprocess_2s.yaml` (2-s windows, 1-s step) produces 149 epochs per recording
  (`results/preprocessing_2s/preprocess_summary.json` → `epochs_per_recording`) with a lower bad-window rate
  (0.066 [`results/preprocessing_2s/preprocess_summary.json` → `final_flag_rates.bad`]).
  **DEVIATION FROM PROPOSAL:** the brief planned to "also test 2 s"; the 2-s epochs were generated but **no
  model was trained or evaluated on them** (time budget, D14) — NOT AVAILABLE.
* **Usable epochs:** overall 0.878 [`results/preprocessing/preprocess_summary.json` → `usable_fraction_overall`]
  of epochs are not "bad". Two recordings are below 50 % usable (sub27_hi, sub37_lo); they are **kept**.
<!-- TABLE:usable -->

## 3.8 Normalisation options and the rest-baseline leakage risk
`src/eegengine/preprocess/normalize.py`:
* `none` — epochs saved un-normalised (default; scaling happens inside each model's training fold).
* `epoch_zscore` — z-score each epoch and channel over time; uses only the epoch itself (no leakage).
* `rest_baseline` (signal level `rest_baseline_signal`, feature level `rest_baseline_features`) — z-score with the
  mean/std of **the same subject's rest (lo) epochs**. Classical experiments use the feature-level version (D9).

**Leakage risk:** for T1 (rest vs task) the test subject's rest recording *is one of the two classes*.
Normalising every feature by that recording makes the rest recording's average feature vector exactly 0 by
construction, so the classifier partly learns "is this the reference recording?". It is a realistic
calibration step (a user records a short baseline before a session) but it inflates T1, so **every T1
result is reported with and without it** and no headline number uses it (§7.3). It uses no labels of other
subjects and never the task recording, so it is not cross-subject leakage.

## 3.9 Before/after PSD
![PSD before/after subject 01](figures/prep_psd_before_after_sub01.png)

*Figure 3.1 — Subject 01: power below 0.5 Hz and above 45 Hz removed; the in-band spectrum is unchanged.*

![PSD before/after subject 38](figures/prep_psd_before_after_sub38.png)

*Figure 3.2 — Subject 38: in the task recording the F3 channel (median-window std 36.98× the median channel,
`results/preprocessing/interpolation_log.csv`) was interpolated, which lowers broadband power after preprocessing.*
