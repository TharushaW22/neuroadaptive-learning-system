# Questions for the other component owners

Status of each answer is updated here; decisions taken meanwhile are in `PLAN.md`.

## Component 1 (EEG engine)

1. **Fold checkpoints.** C3 now retrains per-fold A8 models with your unchanged `run_fold` into
   `component3/artifacts/fold_models/` (D1). Do you agree, or would you rather publish your own
   per-fold checkpoints?
2. **Official integration model.** The API defaults to `T1_xgb_all`; the best deep model is A8.
   Which one should Component 4 receive explanations for?
3. **STEW copy and licence.** Which copy/licence is the team's reference? (C3 uses the copy the
   student placed in `component3/data/raw/STEW/`.)
4. **Regions (D5).** Do you agree with frontal = AF3 F7 F3 FC5 FC6 F4 F8 AF4, temporal = T7 T8,
   parietal = P7 P8, occipital = O1 O2?
5. **Slow content (< 1 Hz).** After the 0.5–45 Hz FIR about two thirds of the cleaned window power
   is still below 1 Hz, and A8 uses it (dropping it flips ~21 % of labels). Is this intended? C3
   explains it as a sixth "slow" band. Would a steeper high-pass (or 1 Hz) change the model?
6. **Team-collected labels.** When will attention/fatigue data arrive?
7. **Checkpoints.** Are the 20261019 checkpoints final? (C3's in-sample results depend on them.)

## Component 2 (computer vision)

Answered by reading C2's code (`docs/C2_ANALYSIS.md`): the model is PyTorch (GRU) plus a scikit-learn
baseline; the input is 15 per-frame facial features × 20 frames (4 s at 5 fps, stride 2 s); the data is
DAiSEE with subject IDs from the folder layout and the official split; there is no fusion model; there
is no probability calibration. Decisions D-V1–D-V5 are in PLAN.md §3b. Open questions (D-V5):

1. **Data and model delivery.** When will DAiSEE be obtained, the features extracted and the GRU trained
   in C2's Python 3.11 environment? Component 3 needs `models/<name>/model.pt + metadata.json` and
   `data/processed/features/*.csv.gz` for **all** splits (docs/VISION_RUNBOOK.md, Part A).
2. **Binary labels.** Please train with `labels.scheme: daisee_binary`, `binary_threshold: 2`
   (low = 0–1, high = 2–3, decision D-V1). Is that acceptable for your own evaluation as well?
3. **Subject-disjoint split.** Does your audit confirm that DAiSEE's official train/validation/test split
   has no participant overlap? Otherwise every vision explanation is labelled IN-SAMPLE.
4. **Subject IDs.** Will the videos keep the official `<split>/<subjectID>/<clipID>/` layout, so that the
   participant IDs are real (not `unknown`)?
5. **Settings.** Will you keep the defaults Component 3 checks: window 20, stride 10, 5 fps, schema 1.0.0,
   GRU? If you change them, please tell us. The readiness check would refuse a mismatch.
6. **Class balance.** DAiSEE engagement is skewed towards high levels. Will you report the binary class
   counts per split, and use class weighting (your default `class_weight: balanced`)?
7. **Library versions.** Which torch/numpy versions will train the model? Component 3 loads the weights in
   Python 3.13 / torch 2.x with `weights_only=True`; the readiness check reports both sides.
8. **Calibration.** C2 has no temperature scaling. Is an uncalibrated P(high) acceptable for Component 4's
   trust indicator, or will you add calibration?

## Component 4 (adaptive policy agent)

1. Is the export enough: per context the recommended explainer, its P(rank 1), whether it is
   practically equivalent to the runner-up; per window the top-k region / band / channel×band
   attributions with a trust level (high / medium / low)? Do you also want natural-language text
   for the LLM prompt?
2. Format: the JSON file (`results/<run>/component4_export.json`, schema in
   `src/xaicomp/export/schema.py`) or the Python functions (`recommendation`,
   `window_explanation`)? A REST endpoint can be added if needed.
3. How will C1's heuristic indices (not in [0, 1]) and the missing "confusion" be mapped into
   `CognitiveState`?
