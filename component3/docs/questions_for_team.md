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

1. Framework and model files?
2. Input representation (frames, landmarks, action units, embeddings) and shape?
3. Classes — the same cognitive states as C1?
4. Time windows — aligned with C1's 4-s / 2-s grid?
5. Dataset — public or own; are subject IDs available?
6. Split method (subject-wise)?
7. Fusion type (early / late / intermediate); differentiable end-to-end?
8. Calibrated probabilities?

C3 is ready for a vision adapter: implement `xaicomp.adapters.base.ModelAdapter` (instances,
interpretable space with groups, value function for coalitions, optional torch module). No C2
data or models are faked meanwhile (D8).

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
