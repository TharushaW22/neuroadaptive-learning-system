# Code guide — what every module does (for the viva)

Each entry: **what** it does, **why** it is needed, **inputs → outputs**, **limitations**.
Paths are relative to `component3/src/xaicomp/` unless stated.

---

## 0. Plumbing

### `_bootstrap.py`
* **What:** adds `../component1/src` to Python's import path and sends all `__pycache__` files to
  `component3/.pycache`.
* **Why:** Component 1 must stay read-only. Installing it with `pip install -e` would write an
  `egg-info` folder into it, and a normal import would write `__pycache__` folders into it.
* **Limitation:** must be imported before `eegengine`; `xaicomp/__init__.py` does this automatically.

### `paths.py` — `safe_path(path)`
* **What:** resolves a path and raises `OutsideComponent3Error` if it is not inside `component3/`.
* **Why:** turns safety rule 1 into code; every file writer calls it.

### `config.py`
* `load_c3_config`, `load_c1_config` (read-only), `load_context`, `run_settings(pilot, run_file)` (merges the
  full budget with the pilot overrides and all other configs; `run_file` = `run.yaml` for EEG,
  `run_vision.yaml` for vision).

---

## 1. Data — `data/eeg_data.py`, `data/instances.py`

### `build_dataset()`
* **What:** loads STEW and runs Component 1's own functions in the same order as C1's
  `run_preprocess.py` (load → filter → bad-channel repair → quality flags → 4-s epochs → labels),
  plus band power and PLV (inputs C1's trainer expects). Saves to `data/processed/`.
* **Why:** C1's script would overwrite C1's committed results; calling the pure library functions
  gives identical data without touching component1/. The config hash (6d7dd612) proves the same settings.
* **Output:** `epochs.npz` (7,104 × 14 × 512), `epochs_meta.parquet`, `continuous.npz` (the 96
  cleaned recordings — needed for the band decomposition), `bandpower.npy`, `adj_plv_mean.npy`.

### `build_manifest(adapter)` / `select_instances(...)`
* **What:** lists every 18-s window (3,264 for T1) with subject, fold, label, the model's
  probability, confidence, share of "bad" epochs and the in-sample flag; then picks N windows per
  subject at random (seeded), balanced between rest and task.
* **Why:** explanations are expensive; the selection must be fair and must never look at the model
  output or at explanations. `select_control_instances` takes a subset of the same windows for the
  negative control so the comparison is paired.

---

## 2. Models — `models/loader.py`, `models/fold_training.py`

* `load_deployed` uses C1's `load_predictor` (read-only) → **in-sample** (C1 trained it on everyone).
* `load_fold` loads a fold model retrained by C3 → **out-of-sample** for the windows of that fold.
* `randomise` builds the same network with freshly initialised weights (negative control).
* `ModelBank` picks fold models if all five exist, else the deployed model, and reports `in_sample`.
* `train_fold_models` repeats C1's fold loop (`deep_runner.py:31-56`) and calls C1's **unchanged**
  `run_fold`; it only adds saving the trained model into `artifacts/`. The done-check compares the
  pooled out-of-fold macro-F1 with C1's reported 0.774 (tolerance ±0.03).
* **Limitation:** CPU training is not bit-identical across machines, hence the tolerance.

---

## 3. Representation — `representation/eeg_bands.py`, `representation/groups.py`

### `decompose(x)`
* **What:** splits each channel into 6 bands (slow < 1, delta 1–4, theta 4–8, alpha 8–13,
  beta 13–30, gamma 30–45 Hz) with a zero-phase FFT filter bank whose weights add up to exactly 1,
  plus a residual (what is left, > 45 Hz).
* **Why:** "remove the alpha rhythm at O1" is a meaningful, explainable feature. Because the parts
  add up exactly to the signal, removing a part and putting the rest back together is well defined.
  It is applied to the whole 150-s recording so overlapping epochs stay consistent.
* **Checks:** `reconstruction_error` (float rounding only), `residual_fraction` (~0.03 % of power).
* **Why the slow band:** after C1's 0.5–45 Hz filter about two thirds of the power is still below
  1 Hz and the model uses it; hiding it in the residual would make every explanation incomplete.

### `InterpretableSpace`
* **What:** the 84 feature names ("O1|alpha") and groupings: channel, band, region (D5),
  region × band. `aggregate()` sums attributions within a group (valid because SHAP and IG
  attributions are additive).

---

## 4. Adapter — `adapters/base.py`, `adapters/eeg_component1.py`

* **What:** the only thing the explainers see. `EEGComponent1Adapter` rebuilds C1's model input
  exactly (8 epochs, divide by channel scale, clip ±20, temperature) — tested to match C1's
  `DeepPredictor.predict` within 1e-5 (measured 5e-8).
* `view(window, k_donors, seed)` → `EEGInstanceView`: the window's components, K donor windows
  from **training subjects of the window's fold**, and `value(masks, replacement)` = the model's
  P(high workload) for many coalitions at once (batched).
* `BandComposedModel` is the differentiable version (for IG): components → sum → epochs → model.
* **Why an adapter:** Component 2 (vision) can be added by writing one new adapter; explainers,
  metrics, statistics, ranking and dashboard stay the same (D8).

---

## 5. Replacement — `replacement/strategies.py`, `replacement/checks.py`

What does "feature absent" mean? Four answers:
* **zero** (off-manifold): the component is set to 0.
* **donor** (marginal): the component comes from another person's window; averaged over K donors.
* **conditional** (Gaussian, D2): fit a multivariate normal to the 84 log band powers of training
  windows (Ledoit-Wolf shrinkage, because 84 dimensions need a stable covariance); for the
  present features use the instance's powers, draw the absent powers from the conditional
  distribution, and rescale the donor waveform to that power.
* **empirical**: use the donors whose band powers are closest on the present features.
* `conditional_power_check` re-measures the band power of the composed signal and reports the
  error vs. the sampled target (pilot: ≈0.013–0.018 log10 units ≈ 0.15 dB).
* **Limitation:** phase and connectivity of the donor waveform are not conditioned.

---

## 6. Explainers — `explainers/`

All return an `Explanation` (84 values, f(x), base value, cost, method-specific checks).
* **KernelSHAP** (`shapley.py`): `shap.KernelExplainer` on the binary coalition space; the
  "model" it calls is our value function with donor replacement. Efficiency (sum of values =
  f(x) − v(empty)) holds exactly — reported as `efficiency_error`.
* **Conditional SHAP**: same estimator and same coalition seed, conditional replacement. Any
  difference to KernelSHAP is caused by the value function only.
* **LIME** (`lime_explainer.py`): random coalitions, exponential kernel on the distance to the
  original, weighted ridge regression (`lime_base.LimeBase`). Keeps the surrogate R².
* **IG** (`integrated_gradients.py`): Captum IntegratedGradients w.r.t. the stacked components;
  summing over time gives one value per channel×band; completeness error reported.
  Donor baseline (main, stochastic) and zero baseline (deterministic, never ranked).

---

## 7. Metrics — `metrics/`

* **fidelity.py:** deletion and insertion curves (remove/add features in the explanation's order,
  5 at a time) for zero and donor removal; compared with random orders (same for all methods) and
  the reverse order. Score = how much better than random, in probability units.
* **stability.py:** mean Spearman correlation between seeds (and top-10 Jaccard).
* **robustness.py:** add noise σ = 0.1 × channel scale, keep only if the label does not flip,
  Spearman between the explanations of the clean and noisy window (same seed).
* **sparsity.py:** Gini index of |attribution| and number of features for 80 % of the mass.
* **agreement.py:** pairwise rank correlation, top-k feature overlap, sign agreement.
* **cost.py:** seconds (single-thread worker) and model evaluations.

---

## 8. Running — `run/orchestrator.py`, `run/tables.py`, `run/analysis.py`

* `run_experiment(pilot)`: for each context builds the manifest and selection, writes
  `run_info.json` (PILOT / IN-SAMPLE / FINAL), then explains windows in parallel worker processes;
  every explanation and every per-window metrics file is saved immediately, so a stopped run resumes.
* `build_tables`: collects the per-window JSON files into tidy parquet tables.
* `analyse_run`: Steps 9–13 (below) + screening table + Component 4 export.

---

## 9. Reliability — `reliability/`

* **test_retest.py:** ICC(2,1) with seeds (or seed pairs, noise draws) as repeated measurements.
* **degradation.py:** mixes explanations with noise (0–100 %) and checks the metric falls
  monotonically (fidelity re-evaluated with the model; the others recomputed).
* **redundancy.py:** Spearman matrix between metrics; |ρ| > 0.9 → merged.
* **randomisation.py:** explanations of the randomised model should differ (method sanity check);
  fidelity should collapse (metric validity).

## 10. Disagreement — `disagreement/`

* **corrected_agreement.py:** ρ_AB / √(ρ_AA ρ_BB) (correction for attenuation); values > 1 flagged;
  IG-zero excluded from correction (deterministic).
* **mixed_effects.py:** `disagreement ~ confidence + LIME R² + band-power correlation + share of
  bad epochs + pair + (1 | subject)` (statsmodels MixedLM); plus the KernelSHAP-vs-Conditional-SHAP
  value-function effect.

## 11. Statistics — `stats/`

* **aggregate.py:** average windows within each subject first (subjects are the independent units).
* **frequentist.py:** Friedman (+ Kendall's W), pairwise Wilcoxon with Holm correction,
  matched-pairs rank-biserial effect size, cluster bootstrap CIs (resample subjects).
* **bayesian.py:** baycomp signed-rank test → P(A better), P(practically the same), P(B better)
  with ROPE widths fixed in `configs/stats.yaml`.

## 12. Ranking — `ranking/`

* **criteria.py:** applies the reliability rules (drop if ICC < 0.5, down-weight if < 0.75, drop
  if not monotonic, drop fidelity if it does not collapse on the control, merge redundant) and
  maps every criterion to a fixed positive scale.
* **topsis.py:** closeness to the ideal method (vectorised).
* **smaa.py:** thousands of draws of Dirichlet(1) weights × bootstrap values → rank
  acceptability, P(rank 1), central weights; repeated with several seeds.

## 13. Export — `export/`

* **schema.py:** JSON schema (version 1.0) — contexts + windows.
* **component4.py:** `recommendation()`, `window_explanation()` (top-k region/band/channel×band +
  trust level from confidence, stability and agreement; in-sample capped at "low"), `build_export()`.

## 14. Dashboard — `component3/dashboard/`

* `common.py`: reading results, colours (Okabe–Ito per method; blue↔orange for attributions;
  status colours with icons), banner, summary/how-to/what-does-this-mean helpers.
* `views_a.py` / `views_b.py`: one function per page, in research order; every page starts with a
  summary generated from the results.
* `app.py`: sidebar sections Overview / EEG / Computer Vision / Multimodal / Cross-context comparison /
  Output for Component 4 / Glossary; one run selector for EEG and one for vision; presentation mode;
  errors become friendly messages.
* `common.py` also holds the modality-aware wording: `set_modality()` + `V(key)`. The EEG strings are the
  original text, so the EEG pages are unchanged and the same page functions draw the vision pages.
* `views_vision.py`: vision Summary / Model & data / Explanations (schematic face coloured by the 5 groups,
  time-segment chart, region bar chart; no real faces because of the DAiSEE licence), the Multimodal
  page (why there is no fusion model), Cross-context comparison, project Overview, Component 4 (both
  contexts). Until vision results exist every vision page shows "Waiting for Component 2's trained model".

---

## 15. Vision context (built ahead, D-V2) — Component 2 is READ-ONLY

### `models/vision_c2.py`
* **What:** minimal **copies** of C2's `TemporalClassifier` (GRU), `Normalizer` (apply only),
  `forward_fill_within_video` and window builder, each citing the C2 file and lines. Also the checkpoint
  checks (`metadata_problems`: real not demo, temporal GRU, `daisee_binary`, 15 features, window 20,
  normaliser present), `select_model_dir` (newest usable checkpoint), `load_vision_model`
  (`torch.load(weights_only=True)`) and `randomise_vision` (negative control).
* **Why copies:** C2's package is called `src` (it would clash on the import path), and importing it
  would compile bytecode inside component2/. `test_step15_vision_c2_copies.py` loads the original files
  read-only and shows the outputs are identical.

### `data/vision_data.py`
* **What:** reads C2's per-clip feature files (only the needed columns; synthetic rows refused),
  rebuilds the windows exactly as C2 does, applies C2's stored training-split normaliser, maps the levels
  0–3 to low/high, and checks whether the split is subject-disjoint (`split_check`). The raw windows are
  cached in `data/processed/vision/`.
* **Output:** `VisionData` (windows, normalised inputs, labels, split check, source).

### `representation/vision_space.py` — `VisionSpace`
* 15 features × S time segments (default 4 × 1 s) = 60 features, in feature-major order (like
  channel × band). Levels: feature_x_segment, feature, segment, group (face regions from the YAML),
  group_x_segment. `space_from_info()` rebuilds the EEG or vision space from `run_info.json`.

### `replacement/vision_strategies.py`
* zero = the training mean (0 after standardisation); donor = frames of a train-split window;
  conditional = Gaussian on the 60 segment means (Ledoit-Wolf), with the donor frames shifted to the
  drawn means; empirical = nearest donors on the present segment means.
* `gaussian_suitability()`: fits on 80 % of the training subjects and checks 90 %-interval coverage on
  the rest, plus the share of discrete dimensions. If the Gaussian fails, Conditional SHAP uses the
  empirical conditional. The numbers are saved in `run_info.json["conditional_check"]`.

### `adapters/vision_component2.py`
* **What:** the vision `ModelAdapter`. Instances = C2 **test-split** windows (fold 0). `view()` gives
  frames (F, S, L), the window mask, K donors from training subjects and `value(masks, replacement)`.
  `VisionGRUModule` is the differentiable path for IG (frames → GRU → P(high)). `in_sample` is True
  unless the split is verified subject-disjoint. Robustness noise = 0.1 training SD, not added to
  blink_event.
* **Limitation:** window labels are copied from the clip (weak labels); the features are geometric
  face measurements, not facial action units.

### `adapters/vision_baseline.py`
* Screening only: C2's scikit-learn baseline (not differentiable → IG screened out). The model file is
  never unpickled.

### `run/vision_readiness.py` + `scripts/check_vision_ready.py`
* Ten read-only checks (model, normaliser, loads here, features, splits/labels, subject IDs, subject
  disjointness, class balance, model vs data). Each check names who must act. Exit code 0 only when
  vision can run. `run_experiment.py --vision` refuses to start otherwise.

### `compare/cross_context.py` + `scripts/compare_contexts.py`
* Puts the finished EEG and vision analyses side by side: P(rank 1), expected rank (Kendall's tau,
  descriptive), reliability decisions per criterion, criterion means, caveats. It compares **method
  suitability per prediction context** (D-V4), never modalities. It returns "not available" until both
  rankings exist.

### Export
* `export/component4.py` adds `modalities` (EEG / vision status: available, available_in_other_run,
  not_available) and `vision_windows` (top-k group / segment / feature / feature × segment). Both keys
  are optional, the schema version stays 1.0 and the EEG `windows` format is unchanged.
