# Vision runbook: when Component 2's model arrives

The vision context is fully built in Component 3 and tested on toy fixtures, but it has never run on
real data, because Component 2 has no DAiSEE data and no trained model yet. This page gives the exact
steps from "Component 2 has trained its model" to "vision results on the dashboard".

Decisions behind it (PLAN.md §3b):

* D-V1: Component 2 trains the model in its own Python 3.11 environment. Engagement is **binary**:
  low = DAiSEE levels 0–1, high = levels 2–3.
* D-V3: there is no multimodal model.
* D-V4: the cross-context comparison compares how suitable each XAI method is in each prediction
  context. It never says that one modality explains a cognitive state better than the other.

Component 3 never writes into `component2/` and never runs Component 2's scripts. Every step in Part A
is done by the **Component 2 owner** in Component 2's own environment.

---

## Part A: what the Component 2 owner must provide

All paths below are relative to `component2/vision_cbu/`. Component 3 reads them and does not change them.

| # | What | Where it must end up | Made by |
|---|---|---|---|
| A1 | DAiSEE videos and label CSVs, in the official layout `DataSet/<Split>/<subjectID>/<clipID>/<clipID>.avi`. Without this layout there are no subject IDs | `data/raw/daisee/videos/`, `data/raw/daisee/annotations/` | C2 owner (licence: research use only) |
| A2 | Per-frame features for **all** splits (train, validation, test), with real participant IDs | `data/processed/features/<clipID>.csv.gz` (+ `features_index.csv`) | `python scripts/audit_dataset.py`, then `python scripts/extract_features.py` |
| A3 | A **binary** engagement GRU trained on the real features (not `--demo`) | `models/<name>/model.pt` + `models/<name>/metadata.json` | `python scripts/train_temporal.py --config <binary config>` (see below) |
| A4 | The normaliser fitted on the training split | inside `metadata.json` (`"normalizer"`). C2 writes it there automatically | same training run |

The binary configuration is a small YAML file that the C2 owner keeps in Component 2. C2's
`load_config` merges it over `configs/default.yaml`:

```yaml
# e.g. component2/vision_cbu/configs/engagement_binary.yaml   (created by the Component 2 owner)
labels:
  target: engagement
  scheme: daisee_binary
  binary_threshold: 2        # low = 0-1, high = 2-3
temporal:
  cell: gru
```

```bash
# in component2/vision_cbu, Component 2's own Python 3.11 environment
python scripts/audit_dataset.py
python scripts/extract_features.py
python scripts/train_temporal.py --config configs/engagement_binary.yaml
python scripts/evaluate.py          # optional: C2's own evaluation report
```

Leave these settings at their defaults, because Component 3 checks them: window 20 frames, stride 10,
5 fps, feature schema 1.0.0 (15 features), `split.use_official_splits: true`.

---

## Part B: what you (Component 3) do

Run everything from `component3/`.

### B1. Readiness check (read-only, about 1–2 min once the features exist)

```bash
.venv/bin/python scripts/check_vision_ready.py
```

Exit code 0 means the vision context can run. Each line is one check. If a check fails, the line names
who must act:

| Check | Fails when | Who |
|---|---|---|
| trained_model | no real temporal GRU with `daisee_binary`, or it is a DEMO model | C2 owner |
| normaliser | `metadata.json` has no training-split normaliser | C2 owner |
| model_loads | the weights do not load into Component 3's GRU copy (the message shows the torch/numpy versions on both sides) | you |
| features | no `*.csv.gz` files, missing columns, or synthetic rows | C2 owner |
| splits_labels | train or test split is missing, or labels fall outside 0–3 | C2 owner |
| subject_ids | test clips have the participant ID `unknown` | C2 owner |
| subject_disjoint | *warning only*: a test participant also appears in train/validation, so every vision result is labelled **IN-SAMPLE** | C2 owner (or accept the label) |
| class_balance | a class has no windows (fail); the minority class is under 20 % (warning) | C2 owner |
| model_vs_data | *warning only*: the checkpoint and the features do not seem to belong together, or test balanced accuracy is near chance | C2 owner |

`run_experiment.py --vision` runs the same check itself and refuses to start until it passes.

### B2. Pilot

```bash
.venv/bin/python scripts/run_experiment.py --vision --pilot      # -> results/vision_pilot/
```

The pilot uses 6 test subjects drawn with a fixed seed, 2 windows each and 2 seeds, with the same small
explainer budgets as the EEG pilot. It also runs the random-weights negative control on 3 subjects.
The first run reads every feature file once and caches the windows in
`data/processed/vision/` (git-ignored). The duration has **not been measured**: the GRU is tiny, so
expect minutes rather than hours. Write the measured time into PROGRESS.md.

### B3. What to check after the pilot

1. `results/vision_pilot/vision__daisee__engagement_binary_gru/run_info.json`:
   * `run_type` is `PILOT` when the split is subject-disjoint, or `PILOT + IN-SAMPLE` when it is not.
   * `split_check` gives the participants per split and any overlap.
   * `conditional_check.mode` is `gaussian` or `empirical`. Record the coverage and discrete share,
     and say which mode was used and why in the write-up.
   * `c2_model` records the checkpoint name, label scheme and library versions.
2. `tables/explanations.parquet`: the `efficiency_error` of KernelSHAP and Conditional SHAP should be
   about 0, and the relative `completeness_error_rel` of IG should be small (< 5 %).
3. Fidelity sanity: on average, the random-order AUC should lie between each method's own order and
   its reverse (`tables/fidelity.parquet`).
4. Negative control (`analysis/control_collapse.parquet`): fidelity should collapse on the
   random-weights GRU.
5. Dashboard (`.venv/bin/streamlit run dashboard/app.py`), section **Computer Vision**: every page
   should show results instead of the "Waiting for Component 2's trained model" message.

### B4. Full run

```bash
.venv/bin/python scripts/run_experiment.py --vision              # -> results/vision_full/
```

The full run uses every test subject, 4 windows each, 3 seeds, 1000 coalitions × 4 donors and IG with
32 steps × 4 donors (the same budgets as EEG, `configs/run_vision.yaml`). It can be resumed: re-run the
same command after an interruption. It refuses to resume into a folder made with another checkpoint or
budget. Estimate the time from the pilot (seconds per window × windows). If it is over 30 minutes, run
it yourself in a terminal, not through Claude.

### B5. Cross-context comparison and Component 4

```bash
.venv/bin/python scripts/compare_contexts.py --eeg-run full --vision-run vision_full
```

This writes `results/cross_context/full__vision_full/comparison.json`. The dashboard page
**Cross-context comparison** appears as soon as both runs have a ranking. The vision run's
`component4_export.json` holds the vision recommendation (`contexts`) and the per-window explanations
(`vision_windows`). In the EEG export, `modalities.vision.status` changes from `not_available` to
`available_in_other_run` the next time the EEG analysis is run (`--stage analyse`).

---

## Troubleshooting

* **The checkpoint uses the 4-class scheme (`daisee_original`).** The check refuses it, because the
  team decided on binary. Ask the C2 owner to retrain with the binary config above.
* **model_loads fails with a version error.** Compare the versions printed by the check. The GRU is
  plain `nn.GRU` + `nn.Linear` and is loaded with `torch.load(weights_only=True)`. If loading still
  fails, report the message (do not install C2's requirements into component3/.venv).
* **The features changed after a run.** The window cache key includes file sizes and times, so a new
  cache is built automatically. Start a new run name for the explanations.
* **Delete the cache:** `rm -r data/processed/vision` (Component 3's own copy only).
