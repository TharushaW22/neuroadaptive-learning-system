# Supervisor demonstration script (about 15 minutes)

**Preparation (before the meeting):** venv active; `python scripts\generate_demo_data.py --train-demo-models --evaluate`
already run; webcam tested with `python scripts\check_environment.py --camera`; good front lighting.

## 1. Context (2 min)
* Research question (see `docs/methodology.md`): can webcam-only visual features, modelled over
  time, estimate DAiSEE-annotated engagement; does GRU/LSTM beat a statistics baseline?
* Scope: computer vision only, no EEG. Dataset not yet downloaded - so today is DEMO MODE.

## 2. Software quality (2 min)
```powershell
python -m pytest -q
```
Point out: all tests run without dataset or webcam; mocked camera; leakage, schema, model-shape,
metrics, privacy tests.

## 3. Safeguards (2 min)
```powershell
python scripts\train_baseline.py    # -> REFUSED (exit 2): no real data
python scripts\evaluate.py          # -> NOT EVALUATED: REAL DATASET NOT AVAILABLE
```
Explain: no fabricated results are possible by construction.

## 4. Dashboard (6 min) - `streamlit run app.py`
1. **Home**: DEMO MODE banner; "Validated model: NO VALIDATED MODEL AVAILABLE".
2. **Live webcam**: start a 20 s capture; show landmarks, EAR, head pose, gaze proxy, quality
   warnings (cover the camera / add a second face / dim the light to trigger UNKNOWN-style warnings).
   State box shows NO VALIDATED MODEL AVAILABLE.
3. **Session analytics**: EAR trace with blinks, head-pose timeline, export CSV/JSON; consent
   checkbox for saving.
4. **Dataset setup & audit**: where files go; run audit -> DATASET_NOT_AVAILABLE.
5. **Model evaluation**: real section says NOT EVALUATED; open the DEMO ONLY expander to show the
   report format (confusion matrices, comparison, ablation) - stress it is synthetic.
6. **Settings & privacy**: no video stored, pseudonyms, deletion.

## 5. Plan once DAiSEE arrives (2 min)
Audit -> extract -> train baseline + GRU -> evaluate -> fill `docs/evaluation_protocol.md` template.
Discuss: label imbalance (macro F1), weak video-level labels, participant-level splits.

## 6. Questions to ask the supervisor (1 min)
* Is `daisee_original` (4-class) the right first target, or binary engagement?
* Is an operational FOCUSED/REDUCED_ATTENTION/DISTRACTED study needed, and with what protocol/ethics approval?
* Acceptable compute budget for full-dataset extraction?
