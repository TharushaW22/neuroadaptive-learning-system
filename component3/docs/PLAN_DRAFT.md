The deployed models were trained on all 48 subjects. If you explain them on STEW, every explained window comes from data the model was trained on.
No per-fold checkpoints were saved. So out-of-fold explanations can't be produced right now.
The STEW data isn't on this machine. component1/data/ doesn't exist (it's git-ignored), and there is no .venv either.
The T3 models are at chance level (AUROC about 0.50). They're only useful as a negative control.
STEP 1: Component 1 (EEG engine)
a) Structure and entry points
component1/ is a pip-installable package (eegengine, in src/). It has one script per pipeline step, all called in order by run_all.py.

Purpose	Entry point
Load STEW	io/stew.py
Preprocess and epoch	preprocess/pipeline.py preprocess_recording(), script run_preprocess.py
Features	features/extract.py extract_features(), script run_features.py
Classical training	run_baselines.py, run_ablation.py, train_final_classical.py
Deep training	run_deep.py (5-fold, plus --final), trainer.py, deep_runner.py
Inference	predictors.py load_predictor(), engine.py CognitiveEngine
API / UI	api/main.py (FastAPI), frontend/ (React)
Evaluation	evaluation/metrics.py, deep_stats.py, audit_leakage.py, export_report.py
⚠️ Running C1's scripts would overwrite C1's committed files (results/…, docs/labels.md). Component 3 must call C1's library functions and write only to its own folder.

b) Dataset
STEW (public, Lim et al. 2018; distributed through IEEE DataPort). Not in the repo; it must be downloaded into data/raw/STEW/, and data/processed/ has to be regenerated.
48 subjects. Two 150-s recordings each: lo (rest) and hi (SIMKAP task). That's 96 recordings at 128 Hz.
14 Emotiv channels, in this order: AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4.
4-s epochs with a 2-s step: 74 per recording, 7,104 in total.
Labels (labels.py):
Task	Definition	Subjects	Epochs
T1 workload_binary	rest = low, task = high	48	3552 / 3552
T2 workload_level	own rating 1–3 / 4–6 / 7–9	45	heavily confounded with T1
T3 workload_within_task	task recordings only; rating 4–6 = moderate, 7–9 = high	45	1480 / 1850
Subjects 5, 24 and 42 have no rating. In T3 each subject contributes only one class.

No labels exist for attention, fatigue or readiness. C1 outputs heuristic band-power ratios for these, not model predictions.
c) Preprocessing and model input
Remove DC, apply a 0.5–45 Hz zero-phase FIR filter and a 50 Hz notch.
Find persistently bad channels and rebuild them by spherical-spline interpolation (at most 3). ICA is off.
Mark each window's quality (good / warn / bad); nothing is dropped.
Cut 4-s / 2-s-step epochs. Epochs are slices of one cleaned continuous recording (pipeline.py:43-80).
Model inputs:

Deep models take raw epochs, not features. One sample is a sequence of L = 8 consecutive epochs, which covers 18 s.
x: (1, 8, 14, 512). Divided by a per-channel channel_scale, then clipped to ±20.
A10 only (GAT branch): node (1, 8, 14, 5), the z-scored log10 band power; and adj (1, 8, 14, 14), PLV averaged over the theta/alpha/beta/gamma bands.
Bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz.
The scalers are stored inside each .pt checkpoint (ck["scalers"]), fit on the training subjects only.
XGBoost uses 1,499 tabular features: spectral 185, entropy 84, wavelet 266, PLV 364, coherence 364, graph 236. The column order is saved in the .joblib (columns) and in results/features/feature_columns.txt. A StandardScaler sits inside the pipeline.
d) Models
All models are PyTorch except the XGBoost ones. Weights are in the repo (component1/models/, under 1 MB each).

Model	File	Architecture	CV result (pooled macro-F1 / AUROC)
T1_A8_cnn_lstm_trans	.pt	EEGNet encoder per epoch, then LSTM and Transformer, then fusion MLP; 130,616 params, T = 1.10	0.774 / 0.835
T1_A10_hybrid	.pt	A8 plus a GAT branch (plain PyTorch)	0.762 / 0.853
T3_A10_hybrid	.pt	same as A10	0.484 / 0.501 (stopped at epoch 0, T = 20, i.e. chance)
T1/T3_xgb_all	.joblib	XGBoost on 1,499 features (not a neural net)	T1 0.865 / 0.941 (LOSO)
Definitions are in hybrid.py, eegnet.py and gat.py.
⚠️ Every deployed checkpoint was trained on all subjects (run_deep.py:93-125, decision D18 in C1's docs/decisions.md).
The per-fold models are not saved. run_fold() returns the model, but run_deep() only writes predictions.
Loading the model and running inference with C1's own code:


from eegengine.inference.predictors import load_predictor
p = load_predictor("T1_A8_cnn_lstm_trans")            # p.model (nn.Module, eval), p.scalers, p.temperature, p.classes
probs = p.predict(buffer)  # buffer: list of 8 dicts {"epoch": (14,512) µV, "bandpower": (14,5), "adj": (14,14)}
# Differentiable path (A8 needs only x):
x = torch.clamp(torch.from_numpy(seq_uV / p.scalers.channel_scale[None,:,None])[None], -20, 20).float()
logits = p.model(x)["workload"] / p.temperature        # (1, 2)
e) Outputs
Real-time: one CognitiveStateOutput JSON every 2 s (output_contract.md). It contains workload.label, calibrated and smoothed probabilities, raw_probabilities, confidence, model_version and task, plus heuristic indices, band power, a PLV matrix and quality flags. It's available over a WebSocket, POST /predict, or the Python engine.
Offline: results/<exp>/predictions.csv.
Deep: subject_id, condition, t_start, t_end, fold, y, p0, raw_p0, p1, raw_p1, with 3,264 sequences for T1 and 1,530 for T3.
Classical: one row per epoch, with quality_flag.
f) Evaluation
Subject-wise splits. LOSO for classical models; a fixed 5-fold subject split (folds_5fold.json) for deep models. There is an inner validation split by subject for early stopping and temperature fitting, and a leakage guard plus a permutation audit (0.506).
Subject IDs are available for every test sample, along with fold, condition and quality.
Results: see the table in (d). T3 is at chance for every model.
g) Dependencies
C1 was developed on Python 3.13 (it requires ≥ 3.11), CPU only.
Packages: numpy, scipy, pandas, pyarrow, mne, PyWavelets, networkx, scikit-learn, xgboost, torch ≥ 2.2, fastapi, pydantic, pytest.
This Mac has Python 3.13.7 but no venv.
h) Existing explainability
None. There's no SHAP, saliency or Captum anywhere in C1.

Synthetic or placeholder data in Component 1
Synthetic data appears only in unit tests:

test_quality.py: synthetic signals with known defects.
test_preprocess.py: random signals.
test_features.py (line 17).
test_models_cv.py: torch.randn shape tests.
None of it reaches the trained models or the results. The dashboard's demo mode replays real engine outputs on STEW. Two things to note, though neither is synthetic:

Attention, fatigue and readiness are heuristic formulas, not predictions.
The T3 models carry no learnable signal.
STEP 2: Component 4 (adaptive policy agent)
a) Inputs. schemas.py CognitiveState takes five floats in [0, 1]: attention, fatigue, confusion, readiness, workload. Next to these sits StudentTraits. Both go through POST /decide (api.py) or PolicyService.decide(). In the Streamlit app the state comes from sliders labelled "SIMULATED — from Components 1 & 2" (app.py:684-690). Rewards and datasets are also simulated (reward_simulator.py, data_generation/).

b) Explanation inputs. There is no XAI input at all: no explanation, trust or attribution field, and no Component 3 mock. C4's own algorithms/topsis.py ranks teaching strategies and has nothing to do with C3's method ranking.

c) How it connects. Only through in-process Python objects or its FastAPI app. There is no file exchange. One mismatch to flag to the C1 and C4 owners: C1's heuristic indices aren't in [0, 1] (fatigue is about 2.0 in the example), and C1 produces no confusion.

STEP 3: Component 2
There's nothing in the repo yet. I found no branch, no folder, no placeholder and no agreed interface (the "fusion" hits were the word "confusion"). origin/main contains only a README. C1's contract mentions "fusion / XAI / cueing" consumers but defines nothing for them.

STEP 4: Plan for Component 3
a) ModelAdapter interface

class ModelAdapter(Protocol):
    context_id: str                    # e.g. "eeg/T1/A8_cnn_lstm_trans"
    modality: str                      # "eeg" | "vision" | "multimodal"
    class_names: list[str]
    space: InterpretableSpace          # feature names (e.g. 70 "O1|alpha"), groupings {"channel": ..., "band": ..., "region": ...}
    def load_instances(self, split) -> InstanceSet       # real samples + subject_id, y, fold, condition, quality
    def background(self, instance) -> Background         # training-subject data of that instance's fold
    def predict_proba(self, X_native) -> np.ndarray      # calibrated (temperature-scaled)
    def compose(self, instance, mask, replacement) -> X_native    # mask over interpretable features -> model input
    def torch_module(self) -> nn.Module | None           # differentiable from interpretable components to probabilities; None -> IG not applicable
    def capabilities(self) -> dict                       # differentiable, n_features, cost/eval, has_subject_ids …
The explainers, metrics and statistics only see this interface.
Adapters are chosen through YAML config (contexts/*.yaml). Vision and fusion adapters will plug in the same way.
The EEG adapter imports C1 through PYTHONPATH=../component1/src rather than pip install -e, because -e writes egg-info into C1's folder.
b) How each explainer fits C1's input
Interpretable features: 70 channel × band components.

Rebuild each 18-s window as a continuous (14 × 2304) segment. This is valid because the epochs are slices of one cleaned recording; overlapping epochs therefore stay consistent when perturbed.
Split it with a zero-phase filter bank into the five bands, giving 70 components, plus a fixed residual (below 1 Hz, filter error). Report the reconstruction error.
Re-slice into 8 epochs, then scale and clip as C1 does.
Group levels: channel (14), band (5), region (proposed: frontal, temporal, parietal, occipital; to be agreed), and channel × band (70).
Method	Application
KernelSHAP	shap.KernelExplainer over a 70-d binary mask. "Absent" means marginal replacement: the component taken from training-subject donors, averaged over K donors.
Conditional SHAP (yours)	Same coalitions. Absent components are drawn from a Gaussian conditional on the 70-d log band-power vector (Ledoit-Wolf covariance fit on the fold's training subjects); the donor waveform is rescaled to the sampled power. Limitation: phase and connectivity aren't conditioned. This is a design proposal for you to confirm.
LIME	lime.lime_base.LimeBase on the same 70-d mask, same replacement as KernelSHAP, kernel width as config.
Integrated Gradients	Captum IntegratedGradients with respect to the stacked band components (5 × 14 × 2304). Since x = Σ components + residual, summing gives channel × band attributions directly and completeness holds. Baselines: (1) zero, meaning the band is absent (off-manifold); (2) training-subject donor components averaged over K (expected-gradients style).
All four methods produce attributions on the same 70 features, so the agreement and fidelity comparisons are like-for-like.

Fidelity (deletion / insertion):
Off-manifold: the component is set to 0.
On-manifold: the donor component or a Gaussian-conditional sample.
Robustness noise: σ = 0.1 × channel scale, matching C1's training augmentation. Only cases where the predicted label doesn't change are scored.
Primary model: T1_A8. It's the best deep T1 model and depends only on x, so it's fully differentiable and cheap (11 ms per sequence).
A10 is optional. Its GAT inputs (band power, PLV) come from the signal, so every perturbation means recomputing them with C1's extract_features (about 0.3 s per sample). That's too expensive for SHAP at scale; IG would also need a differentiable band power and PLV.
XGBoost (optional context): IG would be screened out as not applicable (not differentiable), and the other three run on feature families.
T3: a negative control only. Fidelity should come out near chance.
c) What can run now vs later
On EEG today, everything once data and fold models are in place: screening → explanations → all metrics → reliability → disagreement → statistics → TOPSIS/SMAA → dashboard → C4 export.
Waits for C2: the vision and fusion adapters and their interpretable spaces (e.g. face regions or AU groups; modality → feature groups for fusion), the cross-modality comparison, and the multimodal conflict analysis.
Already config-driven so C2 plugs in without redesign: contexts, explainer parameters, replacement strategies, metric weights.
d) Folder structure and requirements

component 3/
  README.md  docs/PLAN.md  requirements.txt  pyproject.toml  .gitignore (data/, results/_scratch)
  configs/   contexts/eeg_T1_A8.yaml  explainers.yaml  metrics.yaml  stats.yaml  ranking.yaml
  src/xaicomp/
    adapters/      base.py  eeg_component1.py              (vision/fusion added when C2 exists)
    representation/ eeg_bands.py  groups.py
    data/          eeg_data.py (calls C1 library, writes to component 3/data)  instances.py
    screening/     applicability.py
    replacement/   off_manifold.py  donor.py  gaussian_conditional.py
    explainers/    base.py  kernel_shap.py  conditional_shap.py  lime_explainer.py  integrated_gradients.py
    metrics/       fidelity.py  stability.py  robustness.py  sparsity.py  cost.py  agreement.py
    reliability/   test_retest.py  degradation.py  redundancy.py
    disagreement/  corrected_agreement.py  mixed_effects.py
    stats/         aggregate.py  frequentist.py  bayesian.py
    ranking/       topsis.py  smaa.py
    export/        component4.py  schema.py
  scripts/   one per stage   dashboard/app.py (Streamlit)   tests/
  data/ (git-ignored)   results/<context_id>/...
requirements.txt is C1's own list (unchanged lower bounds) plus shap, lime, captum, statsmodels, baycomp, streamlit, all with lower bounds only and no pins. Step 0 confirms they install on Python 3.13; baycomp and lime are old packages but pure Python.

e) Implementation steps (one per session)
#	Step	Done when
0	venv, requirements, C1 importable via PYTHONPATH	load_predictor("T1_A8_cnn_lstm_trans") loads: 130,616 params, T = 1.101, classes [low, high]; git status shows no change in component1/
1	STEW → C3 data dir; epochs and continuous signals via C1 library	7,104 epochs, config hash 6d7dd612, label counts match C1's docs/labels.md
2	Fold models (blocked; see questions)	Out-of-fold predictions reproduce deep_T1_A8…/predictions.csv (pooled macro-F1 0.774)
3	Adapter + band decomposition + torch module	predict_proba equals C1's DeepPredictor.predict (≤ 1e-5); reconstruction error reported; gradients finite
4	Instance manifest + applicability screening	Manifest per subject (fold, y, p, confidence, frac_bad); screening table written
5	Four explainers + group aggregation	On 5 real instances: IG completeness and SHAP efficiency errors small; common output format; cost logged
6	Fidelity (deletion/insertion, two replacement modes)	Curves saved; random-order attribution falls between the two
7	Stability, robustness, sparsity, cost, agreement	Metric table for the 5 instances; IG stability caveat recorded
8	Full resumable run over the instance set	All (instance × method × seed) complete; runtime log
9	Metric reliability	ICC per metric; monotonic response to degradation; redundancy matrix
10	Disagreement	Noise-corrected agreement; MixedLM converges (subject as random effect)
11	Statistics	Subject-level table; Friedman, Wilcoxon-Holm and effect sizes; baycomp with ROPE
12	TOPSIS + SMAA	P(rank 1) per method, stable across Monte-Carlo seeds
13	C4 export	JSON validates against schema
14	Streamlit dashboard	Runs from results/ only
15	C2 adapters	Once C2 has landed
f) Risks and blockers
The deployed models are in-sample on STEW. Fix: per-fold checkpoints, either from C1 or retrained by C3 with C1's unchanged run_fold (same folds and seed) into C3's folder. I need your or C1's approval before doing that.
STEW is missing locally. It needs an IEEE DataPort download (account/licence).
Compute cost.
My rough estimates, from C1's measured 11 ms per sequence: KernelSHAP at about 2,000 evaluations is about 20 s per instance; Conditional SHAP with K = 10 draws is about 10× that.
With roughly 4 instances per subject × 48 subjects × 3 seeds, that's hours, mostly for Conditional SHAP.
Step 5 will measure the real cost before we pick the budget.
IG is deterministic with a fixed baseline. Its "stability" and ρ(A,A) would be trivially 1 and inflate the noise-corrected agreement. Use stochastic baselines, or report IG's stability separately.
The Gaussian assumption on log band power, and no conditioning on phase.
Small samples: 48 subjects for T1. T3 can't support per-subject class metrics.
The A10 / GAT derived-input problem (see b).
Running C1's scripts would overwrite C1's results; pip -e would write into C1.
C2 is absent, and there is no C4 explanation interface.
The folder name component 3 contains a space (it works, but needs quoting in shell commands).
g) Questions for the owners
Component 1

Can you save per-fold checkpoints for A8 (and A10)? Or may I retrain them with your run_fold into my folder?
Which model is the official integration model? The API defaults to T1_xgb_all; the best deep model is A8.
Which STEW copy and licence should I use? Can you share the files?
Do you agree with the region grouping of the 14 Emotiv channels?
When will the team-collected labelled data (attention/fatigue) arrive?
Are the 20261019 checkpoints final?
Component 2

Framework and model files?
What's the input representation (frames, landmarks, AUs, embeddings) and its shape?
Which classes, and are they the same cognitive states as C1?
Time windows: aligned to C1's 4 s / 2 s grid?
Dataset: public or own? Are subject IDs included?
Split method?
Fusion type (early, late or intermediate), and is it differentiable end-to-end?
Calibrated probabilities?
Component 4

What do you need from C3: a recommended explainer per context, a trust score, per-window top-k attributions (channel / band / region), or natural-language text for the LLM prompt?
Format: JSON file, Python function, or REST endpoint?
How will C1's non-[0,1] heuristic indices and the missing confusion be mapped into CognitiveState?
Once you approve, I'll save this as component 3/docs/PLAN.md and rewrite the README's synthetic-data wording to describe the real-data approach. Before step 2 I need two decisions from you:

Whether I may retrain fold models inside the Component 3 folder, if C1 can't provide them.
Whether the Gaussian-conditional design for Conditional SHAP matches what you intended.