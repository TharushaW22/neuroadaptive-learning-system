"""FastAPI backend of the EEG Neuro-Cognitive Understanding Engine.

Run:  .venv\\Scripts\\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000   (from component1/)
      The dashboard reaches it through the Vite proxy at /api (frontend/vite.config.ts), so the browser
      does not need CORS; CORS below only matters when VITE_API_URL points the browser here directly.

REST
  GET  /health                       liveness + available models
  GET  /subjects                     STEW subjects with ratings and labels
  GET  /models                       deployable models (models/*.json)
  GET  /results/summary              summary tables generated from results/ (scripts/export_report.py)
  GET  /results/{experiment_id}      metrics.json of one experiment + its file list
  GET  /results/{experiment_id}/files/{name}   a PNG/CSV of that experiment
  GET  /quality                      per-recording signal quality summary
  GET  /quality/{subject}/{condition} per-window flags of one recording
  GET  /schema                       JSON Schema of CognitiveStateOutput
  GET  /contract/sample              one real output produced by the engine
  POST /predict                      upload a recording/window -> list of CognitiveStateOutput
WebSocket
  WS   /stream/{subject}/{condition}?model=...&speed=1
       replays a STEW recording in real time: every 2 s (divided by speed) it sends
       {"type": "raw", ...} with the new 2 s of filtered signal and {"type": "state", "output": {...}}
"""

from __future__ import annotations

import asyncio
import io
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from eegengine import __version__
from eegengine.config import RESULTS_DIR, load_config
from eegengine.inference.engine import CognitiveEngine
from eegengine.inference.predictors import list_models
from eegengine.inference.schema import CognitiveStateOutput
from eegengine.io.stew import CHANNELS, FS, SUBJECTS, load_ratings, load_recording, rating_to_level
from eegengine.labels import within_task_label

app = FastAPI(title="EEG Neuro-Cognitive Understanding Engine", version=__version__,
              description="J26-DS-333 Component 1 — workload is a model prediction; attention, fatigue and "
                          "readiness are heuristic EEG indices (not trained predictions).")
@app.exception_handler(FileNotFoundError)
async def missing_file(_request, exc: FileNotFoundError):
    """Missing data/model files -> a clear 503 (with CORS headers) instead of a bare 500.

    A bare 500 has no CORS headers, so the browser would misreport it as a CORS error.
    """
    from fastapi.responses import JSONResponse
    name = getattr(exc, "filename", None) or str(exc)
    hint = ("STEW data is missing — put sub01_lo.txt … sub48_hi.txt and ratings.txt in component1/data/raw/STEW/, "
            "then run scripts/run_preprocess.py and scripts/run_features.py. The dashboard's demo mode works without it.")
    return JSONResponse(status_code=503, content={"detail": f"file not found: {name}", "hint": hint})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------------------------- helpers
def default_model() -> str:
    models = list_models()
    if not models:
        raise HTTPException(503, "no deployable models in models/ — run scripts/train_final_classical.py")
    pref = [m for m in models if m["task"] == "workload_binary" and m["kind"] == "classical"]
    return (pref or models)[0]["id"]


def check_model(model: str | None) -> str:
    model = model or default_model()
    if model not in {m["id"] for m in list_models()}:
        raise HTTPException(404, f"unknown model {model!r}")
    return model


@lru_cache(maxsize=1)
def epoch_meta() -> pd.DataFrame:
    from eegengine.data import epochs_paths
    _, mp = epochs_paths(load_config("preprocess.yaml"))
    return pd.read_parquet(mp)


@lru_cache(maxsize=1)
def interpolation_log() -> dict:
    p = RESULTS_DIR / "preprocessing" / "interpolation_log.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {d["recording"]: d for d in json.load(f)}


def read_json(path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found — generate it first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- REST
@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "n_models": len(list_models()), "fs": FS, "channels": CHANNELS}


def subject_ratings() -> dict[int, tuple[float, float]]:
    """Ratings from data/raw/STEW/ratings.txt, or — if the raw data is not present — from the
    committed EDA table results/eda/recordings.csv (same values, written by explore_dataset.py)."""
    try:
        return load_ratings()
    except FileNotFoundError:
        p = RESULTS_DIR / "eda" / "recordings.csv"
        if not p.exists():
            raise
        df = pd.read_csv(p)
        r = df.pivot(index="subject_id", columns="condition", values="rating")
        return {int(s): (float(r.loc[s, "lo"]), float(r.loc[s, "hi"])) for s in r.index}


@app.get("/subjects")
def subjects():
    ratings = subject_ratings()
    out = []
    for s in SUBJECTS:
        lo, hi = ratings[s]
        out.append({
            "subject_id": s,
            "rating_lo": None if np.isnan(lo) else lo, "rating_hi": None if np.isnan(hi) else hi,
            "level_lo": rating_to_level(lo), "level_hi": rating_to_level(hi),
            "within_task_hi": within_task_label("hi", hi), "rated": not np.isnan(hi),
        })
    return out


@app.get("/models")
def models():
    return list_models()


@app.get("/results/summary")
def results_summary():
    return read_json(RESULTS_DIR / "summary.json")


@app.get("/results/{experiment_id}")
def result(experiment_id: str):
    d = RESULTS_DIR / experiment_id
    if not (d / "metrics.json").exists() or ".." in experiment_id:
        raise HTTPException(404, f"unknown experiment {experiment_id!r}")
    r = read_json(d / "metrics.json")
    r["files"] = sorted(p.name for p in d.iterdir() if p.suffix in (".png", ".csv", ".yaml"))
    return r


@app.get("/results/{experiment_id}/files/{name}")
def result_file(experiment_id: str, name: str):
    p = RESULTS_DIR / experiment_id / name
    if ".." in experiment_id or ".." in name or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p)


@app.get("/quality")
def quality():
    m = epoch_meta()
    interp = interpolation_log()
    rows = []
    for (s, c), g in m.groupby(["subject_id", "condition"]):
        flags = g["quality_flag"].value_counts()
        key = f"sub{s:02d}_{c}"
        rows.append({
            "subject_id": int(s), "condition": c, "recording": key, "n_windows": int(len(g)),
            "good": int(flags.get("good", 0)), "warn": int(flags.get("warn", 0)), "bad": int(flags.get("bad", 0)),
            "usable_fraction": float((g["quality_flag"] != "bad").mean()),
            "mean_score": float(g["quality_score"].mean()),
            "interpolated": interp.get(key, {}).get("interpolated", []),
            "interpolation_reasons": {d["channel"]: d["reasons"] for d in interp.get(key, {}).get("detected", [])},
        })
    return rows


@app.get("/quality/{subject}/{condition}")
def quality_windows(subject: int, condition: str):
    m = epoch_meta()
    g = m[(m.subject_id == subject) & (m.condition == condition)]
    if g.empty:
        raise HTTPException(404, "unknown recording")
    return [{"window_index": int(r.epoch_index), "t_start": float(r.t_start), "t_end": float(r.t_end),
             "flag": r.quality_flag, "score": float(r.quality_score),
             "bad_channels": [c for c in str(r.bad_channels).split(";") if c and c != "nan"]}
            for r in g.itertuples()]


@app.get("/schema")
def schema():
    return CognitiveStateOutput.model_json_schema()


@app.get("/contract/sample")
def contract_sample(model: str | None = None):
    model = check_model(model)
    eng = CognitiveEngine(model, participant_id="sub01", session_id="sample")
    outs = eng.run_recording(load_recording(1, "hi").data[:, : 10 * FS])
    return outs[-1].model_dump()


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str | None = Form(None), participant_id: str = Form("P000"),
                  session_id: str | None = Form(None)):
    """Accepts a STEW-format text file (samples x 14, whitespace) or JSON {"data": [[14 x n]]} (channels x samples)."""
    model = check_model(model)
    raw = await file.read()
    try:
        if file.filename and file.filename.endswith(".json"):
            data = np.asarray(json.loads(raw)["data"], dtype=float)
        else:
            data = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None).to_numpy(dtype=float).T
    except Exception as e:  # noqa: BLE001 — report any parse problem to the client
        raise HTTPException(400, f"could not parse upload: {e}") from e
    if data.ndim != 2 or data.shape[0] != len(CHANNELS):
        raise HTTPException(400, f"expected {len(CHANNELS)} channels, got array of shape {list(data.shape)}")
    if data.shape[1] < 4 * FS:
        raise HTTPException(400, "need at least 4 s (512 samples) of data")
    eng = CognitiveEngine(model, participant_id=participant_id, session_id=session_id)
    return [o.model_dump() for o in eng.run_recording(data)]


# ---------------------------------------------------------------------------------------------- WebSocket
@app.websocket("/stream/{subject}/{condition}")
async def stream(ws: WebSocket, subject: int, condition: str, model: str | None = Query(None),
                 speed: float = Query(1.0, gt=0, le=50)):
    await ws.accept()
    try:
        model = check_model(model)
        rec = load_recording(subject, condition)
    except Exception as e:  # noqa: BLE001
        msg = (f"recording sub{subject:02d}_{condition} not found in data/raw/STEW — restore the STEW files "
               "or switch the dashboard to demo mode") if isinstance(e, FileNotFoundError) else str(getattr(e, "detail", e))
        await ws.send_json({"type": "error", "message": msg})
        await ws.close()
        return
    eng = CognitiveEngine(model, participant_id=f"sub{subject:02d}", session_id=f"stew-{subject:02d}-{condition}")
    await ws.send_json({"type": "meta", "subject": subject, "condition": condition, "rating": None if np.isnan(rec.rating) else rec.rating,
                        "model": model, "task": eng.predictor.task, "classes": eng.predictor.classes,
                        "fs": FS, "display_fs": FS // 2, "channels": CHANNELS, "duration_s": rec.duration_s,
                        "speed": speed})
    step = eng.step
    loop = asyncio.get_running_loop()
    try:
        for start in range(0, rec.data.shape[1], step):
            t0 = loop.time()
            chunk = rec.data[:, start:start + step]
            outs = await loop.run_in_executor(None, eng.push, chunk)  # keep the event loop responsive
            for o in outs:
                n_new = eng.win if o.window_index == 0 else step       # first window: all 4 s
                new = eng.last_window[:, -n_new:][:, ::2]             # newest samples, sent at 64 Hz
                await ws.send_json({"type": "raw", "t_start": o.t_end_s - n_new / FS, "fs": FS // 2,
                                    "data": np.round(new, 2).tolist()})
                await ws.send_json({"type": "state", "output": o.model_dump()})
            await asyncio.sleep(max(0.0, step / FS / speed - (loop.time() - t0)))
        await ws.send_json({"type": "end"})
        await ws.close()
    except WebSocketDisconnect:
        return
