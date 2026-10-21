# Evaluation protocol

## Procedure
1. Audit passes (`DATASET_READY`), no unexplained leakage.
2. Extract features once with a fixed configuration; commit the config file (not the data).
3. Train baseline and temporal models on **train**; select hyper-parameters on **validation** only.
4. Run `python scripts\evaluate.py` **once per final configuration** on the **test** split.
   Do not tune on test results.
5. Report window-level and video-level metrics, majority-class reference, confusion matrices,
   curves, latency, ablation and all warnings from the report.
6. Repeat with >= 3 seeds (`project.seed`) for the final configuration and report mean +/- std.

Outputs (`outputs/evaluation/<timestamp>/`): `report.json`, `report.md`, `metrics_summary.csv`,
`per_class_metrics.csv`, `window_predictions.csv`, `ablation.csv`, `confusion_*.png`,
`curves_temporal.png`, `model_comparison.png`. Without data: **NOT EVALUATED: REAL DATASET NOT AVAILABLE**.

Synthetic pipeline checks are written to `data/demo/evaluation/` and every artefact carries
"DEMO ONLY - SYNTHETIC DATA - NOT RESEARCH RESULTS". They must not appear in the thesis results.

---

## Research evaluation template (fill in from a real report)

**Run ID / report folder:** `outputs/evaluation/__________`
**Code version (commit):** ______ **Config file:** ______ **Seeds:** ______

### Data
| Item | Value |
|---|---|
| Dataset | DAiSEE (version/date obtained: ____) |
| Label target / scheme | engagement / daisee_original |
| Video-label policy | broadcast_video_label |
| Split source | official split files |
| Participants (train / val / test) | __ / __ / __ |
| Clips (train / val / test) | __ / __ / __ |
| Excluded clips and reasons | from audit: ____ |
| Windows (train / val / test) | __ / __ / __ |
| Sampling fps / window / stride | __ / __ / __ |
| Face backend | ____ |

### Results (test split)
| Model | Accuracy | Macro P | Macro R | Macro F1 | Video macro F1 | Latency (ms/window) |
|---|---|---|---|---|---|---|
| Majority class (reference) | | - | - | - | - | - |
| Baseline (____) | | | | | | |
| GRU / LSTM | | | | | | |

Per-class table and confusion matrices: insert from report.

### Ablation (baseline, macro F1)
| All features | w/o eye | w/o gaze | w/o head pose | w/o facial |
|---|---|---|---|---|
| | | | | |

### Interpretation checklist
- [ ] Compared against majority-class accuracy (class imbalance)
- [ ] Macro F1 used as primary metric
- [ ] No hyper-parameter chosen on the test set
- [ ] Participant overlap / leakage warnings addressed
- [ ] Claims limited to "estimating DAiSEE-annotated engagement", not attention/distraction
- [ ] Limitations section updated
