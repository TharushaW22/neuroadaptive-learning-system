"""Export an offline snapshot for the dashboard's demo mode (frontend/public/demo/).

It calls the same functions the API serves, so the demo shows real engine outputs and real results.
Usage:  python scripts/export_demo.py [--model T1_A10_hybrid] [--subjects 1 2 16 38]
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so `api` is importable
from api import main as api  # noqa: E402
from eegengine.config import PROJECT_ROOT  # noqa: E402
from eegengine.inference.engine import CognitiveEngine
from eegengine.inference.predictors import list_models
from eegengine.io.stew import CHANNELS, FS, load_recording

OUT = PROJECT_ROOT / "frontend" / "public" / "demo"


def dump(name: str, obj) -> None:
    with open(OUT / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def stream_messages(subject: int, condition: str, model: str) -> list[dict]:
    """The exact message sequence the WebSocket would send (without the real-time waits)."""
    rec = load_recording(subject, condition)
    eng = CognitiveEngine(model, participant_id=f"sub{subject:02d}", session_id=f"stew-{subject:02d}-{condition}")
    msgs = [{"type": "meta", "subject": subject, "condition": condition,
             "rating": None if np.isnan(rec.rating) else rec.rating, "model": model, "task": eng.predictor.task,
             "classes": eng.predictor.classes, "fs": FS, "display_fs": FS // 2, "channels": CHANNELS,
             "duration_s": rec.duration_s, "speed": 1}]
    for start in range(0, rec.data.shape[1], eng.step):
        for o in eng.push(rec.data[:, start:start + eng.step]):
            n_new = eng.win if o.window_index == 0 else eng.step
            msgs.append({"type": "raw", "t_start": o.t_end_s - n_new / FS, "fs": FS // 2,
                         "data": np.round(eng.last_window[:, -n_new:][:, ::2], 1).tolist()})
            out = o.model_dump()
            out["connectivity"]["matrix"] = [[round(v, 3) for v in r] for r in out["connectivity"]["matrix"]]
            msgs.append({"type": "state", "output": out})
    msgs.append({"type": "end"})
    return msgs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 16, 38])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models = list_models()
    model = args.model or api.default_model()
    info = next(m for m in models if m["id"] == model)

    dump("subjects", api.subjects())
    dump("models", [info])                       # demo streams were recorded with this model only
    dump("results_summary", api.results_summary())
    q = api.quality()
    dump("quality", q)
    for r in q:
        dump(f"quality_{r['subject_id']}_{r['condition']}", api.quality_windows(r["subject_id"], r["condition"]))
    dump("schema", api.schema())
    dump("contract_sample", api.contract_sample(model))

    index = []
    for s in args.subjects:
        for c in ("lo", "hi"):
            msgs = stream_messages(s, c, model)
            dump(f"stream_sub{s:02d}_{c}", msgs)
            index.append({"subject": s, "condition": c, "model": model})
            print(f"  stream sub{s:02d}_{c}: {sum(m['type'] == 'state' for m in msgs)} windows")
    dump("stream_index", index)
    size = sum(p.stat().st_size for p in OUT.glob("*.json")) / 1e6
    print(f"demo snapshot -> {OUT} ({size:.1f} MB, model {model})")


if __name__ == "__main__":
    main()
