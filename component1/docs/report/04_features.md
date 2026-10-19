# 4. Feature engineering

Entry point: `src/eegengine/features/extract.py:extract_features(X, fs, ch_names, cfg)`; input `X` of
shape `(N, 14, 512)` µV (N epochs of 4 s at 128 Hz); output a pandas DataFrame `(N, 1499)` plus arrays
`bandpower (N, 14, 5)`, `plv (N, 4, 14, 14)`, `coh (N, 4, 14, 14)`. The **same function** is used for batch
extraction (`scripts/run_features.py`, multi-process via `extract_features_parallel`, 6 workers) and by
the real-time engine on one window at a time, so training and inference features are identical by
construction (`tests` and `scripts/audit_leakage.py:feature_locality` confirm a max difference of
0.0 [`results/leakage_audit.json` → `feature_locality.max_abs_difference`]). Configuration: `configs/features.yaml`.

<!-- BEGIN TABLE:feature_counts -->
| Family (column prefix) | Features | Composition |
|---|---|---|
| spectral | 185 | abs + rel band power (14 ch × 5 bands × 2), 3 ratios × 14 ch, 3 asymmetries |
| entropy | 84 | 6 measures × 14 ch (spectral entropy, Hjorth mobility, Hjorth complexity, log variance, skewness, kurtosis) |
| wavelet | 266 | 6 sub-bands × 14 ch × 3 (log energy, relative energy, std) + 14 wavelet entropies |
| plv | 364 | 91 channel pairs × 4 bands |
| coh | 364 | 91 channel pairs × 4 bands |
| graph | 236 | 4 bands × (4 node metrics × 14 ch + 3 global metrics) |
| **total** | 1499 |  |

Source: `results/features/feature_summary.json`
<!-- END TABLE:feature_counts -->

**Naming convention:** `<family>_<measure>_<band/level>_<channel or pair>`, e.g. `spectral_abs_theta_F3`,
`spectral_ratio_theta_beta_AF4`, `entropy_hjorth_mobility_O1`, `wavelet_relenergy_D4_P7`,
`plv_alpha_AF3-F7`, `coh_beta_O1-O2`, `graph_alpha_clustering_T8`, `graph_theta_efficiency`. Experiments select
families by prefix (`extract.py:family_columns`). The full list of 1,499 names is in
`results/features/feature_columns.txt`.

**Caching:** the feature table is saved as `data/processed/features_<hash>.parquet` and the arrays as
`features_<hash>_arrays.npz`, where the hash covers the feature configuration *and* the preprocessing hash
(`src/eegengine/data.py:feature_hash`). `scripts/run_features.py` skips extraction if the file exists.

<!-- BEGIN TABLE:feature_timing -->
| Step | CPU seconds (all 7104 epochs, summed over workers) |
|---|---|
| spectral | 4.8 |
| entropy | 7.4 |
| wavelet | 1.9 |
| connectivity | 85.4 |
| graph | 143.3 |

Wall time 51.7 s; single-window extraction latency (median of 20) 24.47 ms. Source: `results/features/feature_summary.json`
<!-- END TABLE:feature_timing -->

Frequency bands (`src/eegengine/features/spectral.py:BANDS`): delta 1–4, theta 4–8, alpha 8–13, beta 13–30,
gamma 30–45 Hz. Bins are assigned half-open [lo, hi) so adjacent bands never double count; the bands tile
1–45 Hz exactly.

## 4.1 Spectral features (prefix `spectral_`, 185 features)
**WHAT.** The Welch power spectral density averages periodograms of overlapping, windowed segments:
`P(f) = (1/K) Σ_k |FFT(w · x_k)(f)|² / (fs Σ w²)`. Band power is the integral of the PSD over the band:
`BP_b = Σ_{f∈b} P(f) Δf` (µV²). From it:
* **absolute band power** `log10(BP_b)` per channel and band (log because band power is strongly right-skewed),
* **relative band power** `BP_b / Σ_b BP_b` (fraction of 1–45 Hz power; the five values sum to 1 — tested in
  `tests/test_features.py:test_relative_band_powers_sum_to_one`),
* **ratios** `log10(BP_θ/BP_α)`, `log10(BP_θ/BP_β)`, `log10(BP_α/BP_β)` per channel,
* **frontal alpha asymmetry** `ln BP_α(right) − ln BP_α(left)` for F4−F3, AF4−AF3, F8−F7.

**Physiology.** Frontal-midline **theta** (4–8 Hz) rises with working-memory load and mental effort;
**alpha** (8–13 Hz) is high when a region is idle or actively inhibited and is suppressed by visual
attention and task engagement (posterior alpha drops when the eyes take in a task); **beta** (13–30 Hz)
accompanies active, alert processing and motor preparation; **gamma** (30–45 Hz) on a consumer headset is
heavily mixed with muscle (EMG) activity. The theta/alpha and theta/beta ratios are classic workload
markers; frontal alpha asymmetry relates to approach/withdrawal motivation.

**WHY.** Band power is the most established and interpretable EEG workload feature and the A1 baseline of
the ablation. Welch (vs a single periodogram) reduces variance; vs multitaper it is simpler and standard.

**HOW.** `spectral.py:welch_psd` (`scipy.signal.welch`, nperseg 256 [`configs/features.yaml` → `welch.nperseg`]
samples = 2 s, noverlap 128 [`configs/features.yaml` → `welch.noverlap`] = 50 %, Hann window, 0.5 Hz
resolution, 3 segments per 4-s epoch), `spectral.py:band_power`, `spectral.py:spectral_features`.
Input `(N, 14, 512)` → PSD `(N, 14, 257)` → band power `(N, 14, 5)`.

**PRODUCED.** A1 (spectral only) reaches T1 pooled macro-F1 0.810 [`results/cls_T1_xgb_A1_spectral_none_loso/metrics.json` → `pooled.macro_f1`] (§5.4).

**LIMITATIONS.** Absolute power depends on skull/electrode impedance and varies strongly between people;
gamma is contaminated by EMG; a 4-s epoch gives only 3 Welch segments (noisy estimates).

## 4.2 Complexity / time-domain features (prefix `entropy_`, 84 features)
**WHAT** (per channel; `src/eegengine/features/entropy.py`):
* **Spectral entropy** — Shannon entropy of the normalised 1–45 Hz PSD, divided by its maximum:
  `H = −Σ p_f log p_f / log N_f`, `p_f = P(f)/Σ P(f)`; 1 = flat/white spectrum, low = power concentrated in
  few frequencies (`spectral_entropy`). Tested: white noise > 0.9, pure sine < 0.3.
* **Hjorth parameters** (`hjorth`): activity `= var(x)`; mobility `= sqrt(var(x′)/var(x))` (≈ mean frequency;
  tested: for a sine ≈ 2πf/fs); complexity `= mobility(x′)/mobility(x)` (≈ bandwidth / irregularity), with x′ the
  first difference. Activity equals variance, so it is stored once as `entropy_logvar` (`log10 var`).
* **Statistical moments:** skewness and excess kurtosis (`scipy.stats`). Kurtosis is high for spiky signals
  (blinks, pops).

**WHY.** Cheap, interpretable signal-complexity descriptors complementary to band power; mental effort
tends to change the irregularity of EEG.

**PRODUCED.** Adding them to spectral (A2) gives T1 pooled macro-F1 0.826 [`results/cls_T1_xgb_A2_spectral_entropy_none_loso/metrics.json` → `pooled.macro_f1`]
— a small but significant gain over A1 (Wilcoxon p = 0.0028 [`results/stats_classical.json` → `tests.0.p_value`]).

**LIMITATIONS.** Kurtosis/skewness react strongly to artefacts; spectral entropy depends on the band limits.
No sample/permutation entropy was implemented (NOT AVAILABLE).

## 4.3 Discrete wavelet transform (prefix `wavelet_`, 266 features)
**WHAT.** A 5-level DWT with the Daubechies-4 (db4) wavelet splits each epoch into one approximation and five
detail sub-bands. At 128 Hz they correspond approximately to: D1 32–64 Hz, D2 16–32 Hz, D3 8–16 Hz,
D4 4–8 Hz, D5 2–4 Hz, A5 0–2 Hz. Per sub-band and channel: **energy** `log10 Σ c²`, **relative energy**
`E_j / Σ_j E_j`, **std** of the coefficients; per channel **wavelet entropy** `−Σ_j p_j ln p_j` with p_j the
relative energies (`src/eegengine/features/wavelet.py:wavelet_features`, `pywt.wavedec(..., "db4", level=5)`).

**WHY db4.** db4 has a compact support (8 taps) and a shape similar to EEG transients, and is the most
common choice in EEG DWT studies; 5 levels at 128 Hz map neatly onto the classical bands. The DWT gives a
time-frequency view complementary to Welch (better time localisation, octave-wide bands).

**PRODUCED.** A3 (spectral + wavelet) T1 pooled macro-F1 0.822 [`results/cls_T1_xgb_A3_spectral_wavelet_none_loso/metrics.json` → `pooled.macro_f1`],
not significantly better than A1 (p = 0.107 [`results/stats_classical.json` → `tests.1.p_value`]) — largely redundant
with band power.

**LIMITATIONS.** Octave bands do not align exactly with the clinical bands (e.g. D3 8–16 Hz mixes alpha and
low beta); D1 (32–64 Hz) is mostly above the Emotiv's 43 Hz on-board low-pass.

## 4.4 Phase-locking value (prefix `plv_`, 364 features)
**WHAT.** For each band, every epoch is band-pass filtered (4th-order zero-phase Butterworth,
`connectivity.py:band_filter`), the instantaneous phase φ(t) is taken from the analytic signal (Hilbert
transform, `scipy.signal.hilbert`), and for every channel pair
`PLV_ij = | (1/T) Σ_t exp(i (φ_i(t) − φ_j(t))) |`.
PLV = 1 means a constant phase difference (phase-locked), 0 means random phase relations.
Implementation: `connectivity.py:plv_matrix` computes all pairs at once as `|Z Zᴴ| / T` with
`Z = exp(iφ)`. Bands: theta, alpha, beta, gamma (`configs/features.yaml` → `connectivity.bands`).
Output `(N, 4, 14, 14)` symmetric matrices with 1 on the diagonal; the 91 upper-triangle values per band
are the classical features; the matrices feed the GAT and the dashboard.

**WHAT IT MEASURES.** Functional coupling — whether two regions oscillate in a consistent timing
relationship, independent of amplitude. Workload changes fronto-parietal theta/alpha synchronisation.

**Tested:** identical signals → 1; constant phase lag → > 0.99; independent noise → < 0.1; symmetric with unit diagonal.

**Volume-conduction caveat.** One brain source is picked up by several nearby electrodes at the same time
(zero phase lag), which inflates PLV between neighbours without any real interaction. Measures that ignore
zero-lag coupling (imaginary coherence, weighted phase-lag index) avoid this but were not implemented
(NOT AVAILABLE). Long-range edges are therefore more informative than neighbouring ones.

## 4.5 Magnitude-squared coherence (prefix `coh_`, 364 features)
**WHAT.** `C_ij(f) = |S_ij(f)|² / (S_ii(f) S_jj(f))`, where S are Welch cross-/auto-spectra (Hann window,
1-s segments = coh_nperseg 128 [`configs/features.yaml` → `connectivity.coh_nperseg`] samples, 50 % overlap → 7
segments per 4-s epoch), averaged over the bins of each band. Implemented vectorised for all pairs at
once in `connectivity.py:coherence_matrix` (chunks of 256 epochs to bound memory) and tested against
`scipy.signal.coherence` (difference < 1e-6).

**PLV vs coherence.** Coherence is a frequency-domain correlation that depends on **both amplitude and
phase** consistency across segments; PLV uses **phase only** within the epoch. Large-amplitude artefacts
can dominate coherence; PLV is amplitude-independent but needs band filtering. Using both gives the
models two partly independent views of coupling.

**PRODUCED.** Connectivity (PLV + coherence) is the family with the largest gain: A4 T1 pooled macro-F1
0.866 [`results/cls_T1_xgb_A4_spectral_connectivity_none_loso/metrics.json` → `pooled.macro_f1`] vs A1 0.810, Wilcoxon
p = 5.4e-05 [`results/stats_classical.json` → `tests.2.p_value`] (Holm-corrected 1.7e-04 [`results/stats_classical.json` → `tests.2.p_holm`]).

**LIMITATIONS.** Same volume-conduction issue; only 7 segments per epoch → positively biased estimates
for short data.

## 4.6 Graph features (prefix `graph_`, 236 features)
**WHAT.** Each PLV matrix (per epoch, per band) is treated as a weighted graph: 14 electrodes = nodes,
PLV = edge weight (`src/eegengine/features/graph.py`, networkx).

**Thresholding** (`graph.py:threshold_matrix`): three strategies are implemented — *proportional* (keep the
strongest fraction of edges), *absolute* (keep PLV ≥ a value) and *none* (fully weighted). Used:
**proportional, 0.3 [`configs/features.yaml` → `graph.proportion`]** (top 30 % of the 91 edges = 27 edges, weights
kept). **Why:** absolute thresholds give graphs of very different density for different people/bands
(PLV levels differ), which makes metrics incomparable; proportional thresholding fixes the density so
metrics describe the *structure* of the strongest connections.

Metrics (`graph.py:graph_metrics`), with `w_ij` the kept weights:
| Metric | Formula / definition | Meaning |
|---|---|---|
| node strength | `s_i = Σ_j w_ij` | how strongly a region is coupled overall |
| clustering coefficient (weighted, Onnela) | `C_i = (1/(k_i(k_i−1))) Σ_jk (ŵ_ij ŵ_jk ŵ_ki)^{1/3}`, ŵ = w / max w | how much a node's neighbours are also coupled (local segregation) |
| betweenness centrality | fraction of shortest paths (distance `d_ij = 1/w_ij`) passing through the node | hub / bridge role |
| eigenvector centrality | leading eigenvector of W (numpy `eigh`), normalised | connected to other well-connected nodes |
| global efficiency (weighted) | `E = (1/(n(n−1))) Σ_{i≠j} 1/d_ij*`, d* = shortest-path distance (scipy Dijkstra), 1/∞ = 0 | network integration |
| density | fraction of the 91 possible edges present (for "none": mean weight) | constant 0.3 under proportional thresholding |
| graph entropy | `−Σ_i p_i log p_i / log n`, `p_i = s_i/Σ s` | how evenly coupling is spread over nodes |

4 node metrics × 14 channels + 3 global metrics = 59 per band × 4 bands = 236 features.
Tested: complete graph → strength 13, clustering 1, betweenness 0, efficiency 1, entropy 1; a disconnected
graph with one edge → efficiency 1/91; proportional threshold keeps exactly round(0.3 × 91) edges.

**WHY.** Graph metrics summarise the whole network in a few interpretable numbers (integration,
segregation, hubs), complementing the raw 91 pairwise values.

**PRODUCED.** A5 (spectral + connectivity + graph) T1 pooled macro-F1 0.870 [`results/cls_T1_xgb_A5_spectral_connectivity_graph_none_loso/metrics.json` → `pooled.macro_f1`]
— practically the same as A4 (graph metrics are derived from PLV, so they add little new information).
Graph extraction is the slowest step (143.3 [`results/features/feature_summary.json` → `cpu_seconds_by_family.graph`] CPU-s).

**LIMITATIONS.** With 14 nodes, path-based metrics (betweenness, efficiency) are coarse; density is
constant by design under proportional thresholding (a zero-information feature); inherits PLV's
volume-conduction bias.

## 4.7 Real-time cost
Extracting all families for a single 4-s window takes a median of 24.47 [`results/features/feature_summary.json` → `single_epoch_latency_ms_median`] ms
in the latest run of `scripts/run_features.py` (earlier runs on the same machine measured about 40 ms,
the value recorded inside the classical experiments' latency fields — the machine was shared with other
applications, so latency varies between runs).
