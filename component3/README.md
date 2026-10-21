# Component 3 — Comparative Explainable AI (XAI) evaluation

Part of the Neuro-Adaptive Intelligent Learning System (J26-DS-333). Component 3 answers:
**which explanation method can we trust for the system's cognitive-state predictions?**

It explains the **real** EEG workload model of Component 1 on the **real** STEW dataset — no
synthetic data — with four methods (KernelSHAP, Conditional SHAP, LIME, Integrated Gradients),
measures the quality of every explanation, checks that those quality metrics are themselves
reliable, compares the methods statistically at the subject level, ranks them without fixing
anyone's preferred weights (TOPSIS + SMAA), and hands the recommendation to Component 4.

## How it works

1. **Data** — STEW (48 people, rest vs. a demanding task, 14 EEG channels) is cleaned and cut into
   windows with Component 1's own code. Component 1 is only read, never changed.
2. **Model** — Component 1's CNN-LSTM-Transformer (`T1_A8`). Per-fold copies are retrained so each
   explained person was never seen by the model (until then results are labelled IN-SAMPLE).
3. **Features** — each 18-s window is split into 14 channels × 6 frequency bands = 84 parts.
4. **Explanations** — every method scores the same 84 parts, so comparisons are like-for-like.
5. **Quality** — fidelity (deletion/insertion), stability, robustness, sparsity, cost, agreement.
6. **Trust in the metrics** — test-retest ICC, response to deliberately degraded explanations,
   redundancy, and a negative control (the model with random weights).
7. **Statistics** — subject level: Friedman, Wilcoxon-Holm, effect sizes, bootstrap CIs,
   Bayesian signed-rank with a region of practical equivalence (ROPE).
8. **Ranking** — TOPSIS under thousands of random weightings (SMAA) → chance of ranking first.
9. **Output** — a JSON export + Python functions for Component 4, and a dashboard for everyone.

## Vision context (Component 2) — built ahead, waiting for the trained model

The same study is ready for Component 2's vision engagement model: a GRU on 15 facial measurements per
frame over 4-s windows from DAiSEE, with binary engagement (low = 0–1, high = 2–3). The Component 2
owner trains it in Component 2's own environment. Component 3 only reads the trained model, its
normaliser and the extracted features. As soon as they exist:

```bash
.venv/bin/python scripts/check_vision_ready.py            # what is missing, and who must provide it
.venv/bin/python scripts/run_experiment.py --vision --pilot
.venv/bin/python scripts/run_experiment.py --vision
.venv/bin/python scripts/compare_contexts.py --eeg-run full --vision-run vision_full
```
Step by step: `docs/VISION_RUNBOOK.md`. The two contexts are compared only on **how suitable each
explanation method is** in each prediction context, never on which modality explains a state better.
There is **no multimodal model**: the project has no EEG and video recorded from the same learners at the
same time (the dashboard's Multimodal page explains this).

## Setup (macOS / Linux, Python 3.13)

```bash
cd component3
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
source env.sh            # optional: PYTHONPATH for Component 1 (scripts do this themselves)
```
Windows: `py -3.13 -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`, `. .\env.ps1`.

Place STEW in `component3/data/raw/STEW/` (`sub01_lo.txt` … `sub48_hi.txt`, `ratings.txt`), then:

```bash
.venv/bin/python scripts/build_data.py                         # ~1 min -> data/processed/
.venv/bin/python -m pytest                                     # tests (add -m "not slow" for a quick run)
.venv/bin/python scripts/train_fold_models.py --context eeg_T1_A8   # fold models (~30–45 min)
.venv/bin/python scripts/run_experiment.py --pilot             # pilot (~15-30 min)
.venv/bin/python scripts/run_experiment.py                     # full experiment (~10-11 h on an M1; resumable)
.venv/bin/streamlit run dashboard/app.py                       # dashboard
```

The full run can be stopped and restarted at any time: finished explanations are never recomputed.

## Folders

`configs/` all settings · `src/xaicomp/` the library · `scripts/` entry points · `dashboard/` ·
`tests/` · `docs/` (PLAN, PROGRESS, CODE_GUIDE, VISION_RUNBOOK, C2_ANALYSIS, questions_for_team, ai_use_log) ·
`data/`, `artifacts/` (git-ignored) · `results/<run>/` outputs.

## Honesty notes

* Results from the deployed model are **IN-SAMPLE** (it was trained on every subject) and are
  labelled so everywhere; final results need the fold models.
* PILOT runs use small budgets and are for checking the pipeline only.
* Vision results do not exist yet: every vision page says "Waiting for Component 2's trained model".
  Vision results will be labelled IN-SAMPLE unless Component 2's split is verified subject-disjoint.
* No toy or generated data is ever used as a result; toy fixtures live only in `tests/`.
