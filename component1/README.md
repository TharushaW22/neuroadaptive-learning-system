# EEG Neuro-Cognitive Understanding Engine (J26-DS-333, Component 1)

Part of *Neuroadaptive Learning System Using EEG, Computer Vision, and Agentic AI* (SLIIT IT4010).

A reproducible Python engine that turns 14-channel Emotiv EEG into calibrated cognitive-state
estimates, with a FastAPI backend and a React dashboard.

> **Status:** Phase 1 complete: repo setup, STEW loader, signal quality checks, EDA.
> Later sections are placeholders that get filled in as each phase lands.

## Pipeline
acquisition → signal quality → preprocessing → epoching → features → models → calibration → JSON output → dashboard

## Dataset
STEW (Simultaneous Task EEG Workload): 48 subjects × (rest, SIMKAP task), 2.5 min each,
14 channels @ 128 Hz. Place the text files in `data/raw/STEW/` (`sub01_lo.txt` … `sub48_hi.txt`, `ratings.txt`).

**Label honesty:** STEW only provides *workload* labels (rest/task and a 1–9 self-rating).
Attention, fatigue and learning readiness are **not** trained predictions in this version.

## Setup (Windows)
```powershell
cd component1
py -3.13 -m venv .venv            # 3.11+ works
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```
Linux/macOS: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`

## Commands
| Step | Command | Output |
|---|---|---|
| Tests | `python -m pytest` | |
| Dataset EDA | `python scripts/explore_dataset.py` | `results/eda/` |
| Signal quality report | `python scripts/run_quality.py` | `results/quality/` |

## Repository layout
See [CLAUDE.md](CLAUDE.md) for the layout and conventions.

## Results
_To be generated from `results/` in later phases._

## Using team-collected Emotiv data
_To be written in Phase 8._
