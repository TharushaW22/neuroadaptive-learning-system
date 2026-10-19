"""Output contract of the EEG engine (consumed by the fusion, XAI and cueing components).

One CognitiveStateOutput is produced per analysis window (4 s, every 2 s). Only `workload` is a
trained model prediction (source = "model"). Attention, fatigue and readiness are established EEG
band-power indices (source = "heuristic_index") because STEW has no labels for them — consumers
must not treat them as validated predictions. See docs/output_contract.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"
HEURISTIC_NOTE = "heuristic index — not a trained prediction"


class SignalQuality(BaseModel):
    overall: float = Field(..., ge=0, le=1, description="1 - fraction of bad channels in this window")
    flag: Literal["good", "warn", "bad"] = Field(..., description="good: no bad channel; warn: blinks or <=2 bad; bad: more")
    bad_channels: list[str] = Field(default_factory=list, description="channels failing a check in this window")
    interpolated_channels: list[str] = Field(default_factory=list, description="persistently bad channels rebuilt by spherical-spline interpolation")


class WorkloadState(BaseModel):
    label: str = Field(..., description="argmax class of the smoothed probabilities")
    probabilities: dict[str, float] = Field(..., description="calibrated, EMA-smoothed class probabilities (sum to 1)")
    raw_probabilities: dict[str, float] = Field(..., description="calibrated probabilities of this window only (no smoothing)")
    confidence: float = Field(..., ge=0, le=1, description="max smoothed probability")
    source: Literal["model"] = "model"
    model_version: str
    task: str = Field(..., description="workload_binary (low/high) or workload_within_task (moderate/high)")


class HeuristicState(BaseModel):
    value: float = Field(..., description="EMA-smoothed index value")
    raw_value: float = Field(..., description="value of this window only")
    source: Literal["heuristic_index"] = "heuristic_index"
    formula: str
    note: str = HEURISTIC_NOTE


class States(BaseModel):
    workload: WorkloadState
    attention: HeuristicState
    fatigue: HeuristicState
    readiness: HeuristicState


class Connectivity(BaseModel):
    metric: Literal["plv", "coh"] = "plv"
    band: str
    channels: list[str]
    matrix: list[list[float]] = Field(..., description="symmetric channels x channels matrix, 1 on the diagonal")


class CognitiveStateOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    participant_id: str
    session_id: str
    window_index: int = Field(..., ge=0)
    timestamp_start: str = Field(..., description="ISO 8601 UTC start of the analysis window")
    timestamp_end: str = Field(..., description="ISO 8601 UTC end of the analysis window")
    t_start_s: float = Field(..., description="seconds since the start of the session")
    t_end_s: float
    signal_quality: SignalQuality
    states: States
    band_power: dict[str, dict[str, float]] = Field(..., description="channel -> band -> absolute power (uV^2)")
    band_power_unit: str = "uV^2"
    connectivity: Connectivity
    latency_ms: float = Field(..., description="processing time of this window (quality + preprocessing + features + model)")
