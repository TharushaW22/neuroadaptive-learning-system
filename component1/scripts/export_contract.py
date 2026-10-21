"""Export the output contract for teammates: JSON Schema + one REAL example produced by the engine.

Usage:  python scripts/export_contract.py [--model T1_xgb_all]
Writes: docs/output_contract.schema.json, docs/output_example.json
"""

from __future__ import annotations

import argparse
import json
import sys

from eegengine.config import PROJECT_ROOT
from eegengine.inference.engine import CognitiveEngine
from eegengine.inference.predictors import list_models
from eegengine.inference.schema import CognitiveStateOutput
from eegengine.io.stew import FS, load_recording


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    docs = PROJECT_ROOT / "docs"
    schema = CognitiveStateOutput.model_json_schema()
    schema["$id"] = "https://j26-ds-333/eeg-engine/cognitive-state-output.schema.json"
    with open(docs / "output_contract.schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    model = args.model or next(m["id"] for m in list_models() if m["task"] == "workload_binary")
    eng = CognitiveEngine(model, participant_id="sub01", session_id="example-sub01-hi")
    outs = eng.run_recording(load_recording(1, "hi").data[:, : 20 * FS])
    with open(docs / "output_example.json", "w", encoding="utf-8") as f:
        json.dump(outs[-1].model_dump(), f, indent=2)
    print(f"schema + example written (model {model}, {len(outs)} windows, last latency {outs[-1].latency_ms:.1f} ms)")


if __name__ == "__main__":
    main()
