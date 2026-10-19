"""Real-time Cognitive Engine: raw 14-channel EEG in, CognitiveStateOutput per window out.

For every new 4 s window (one every 2 s, the same grid as the training epochs):
  1. preprocessing   — the last `context_s` seconds of raw signal are mean-removed, FIR band-passed
                       (0.5-45 Hz) and notch-filtered, and the newest 4 s are kept. Filtering a
                       longer context avoids filter edge effects inside the window.
  2. signal quality  — the Phase 2 checks on that window. A channel that is hard-bad in >= 60 %
                       of the recent windows is treated as persistently bad and interpolated
                       (spherical spline), as in offline preprocessing.
  3. features        — the same `extract_features` used for training (band power, PLV, ...)
  4. model           — classical (one window) or deep (sequence of the last 8 windows; fewer at start)
  5. smoothing       — exponential moving average of probabilities and indices:
                       s_t = alpha * x_t + (1 - alpha) * s_{t-1}
  6. output          — pydantic CognitiveStateOutput (docs/output_contract.md)
Known difference to offline training: offline epochs come from filtering the whole 150 s recording
at once; online, each window is filtered with 20 s of past context. tests/test_engine.py checks
that the resulting band powers agree closely.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from collections import deque

import numpy as np

from eegengine.config import load_config
from eegengine.features.extract import extract_features
from eegengine.features.spectral import BANDS
from eegengine.inference.heuristics import FORMULAS, heuristic_indices
from eegengine.inference.predictors import load_predictor
from eegengine.inference.schema import (CognitiveStateOutput, Connectivity, HeuristicState, SignalQuality, States,
                                        WorkloadState)
from eegengine.io.stew import CHANNELS, FS
from eegengine.preprocess.bad_channels import interpolate_channels
from eegengine.preprocess.filters import filter_recording
from eegengine.preprocess.pipeline import epoch_quality_config
from eegengine.quality.checks import assess_window


class CognitiveEngine:
    def __init__(self, model: str = "T1_xgb_all", participant_id: str = "P000", session_id: str | None = None,
                 ema_alpha: float = 0.4, session_start: dt.datetime | None = None, context_s: float = 20.0,
                 connectivity_band: str = "alpha", preprocess_config: str = "preprocess.yaml",
                 features_config: str = "features.yaml", ch_names: list[str] | None = None, fs: int = FS,
                 threads: int = 1):
        self.pcfg = load_config(preprocess_config)
        self.fcfg = load_config(features_config)
        self.qcfg = epoch_quality_config(self.pcfg)
        self.predictor = load_predictor(model, threads)
        self.participant_id = participant_id
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.session_start = session_start or dt.datetime.now(dt.timezone.utc)
        self.alpha = ema_alpha
        self.fs = fs
        self.ch = list(ch_names or CHANNELS)
        self.win = int(self.pcfg["epoching"]["window_s"] * fs)
        self.step = int(self.pcfg["epoching"]["step_s"] * fs)
        self.context = int(context_s * fs)
        self.band = connectivity_band
        self.conn_bands = self.fcfg["connectivity"]["bands"]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.buffer = np.zeros((len(self.ch), 0))   # raw samples not yet discarded
        self.buffer_offset = 0                       # absolute sample index of buffer[:, 0]
        self.n_seen = 0                              # total samples pushed
        self.next_end = self.win                     # absolute end sample of the next window
        self.window_index = 0
        self.history = deque(maxlen=self.pcfg["bad_channels"].get("history_windows", 30))
        self.seq = deque(maxlen=max(1, self.predictor.seq_len))
        self.ema_p = None
        self.ema_h = None

    # ------------------------------------------------------------------ input
    def push(self, chunk: np.ndarray) -> list[CognitiveStateOutput]:
        """Append raw samples (channels x n, microvolts, DC offset allowed). Returns new outputs."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape[0] != len(self.ch):
            raise ValueError(f"expected {len(self.ch)} channels, got {chunk.shape[0]}")
        self.buffer = np.concatenate([self.buffer, chunk], axis=1)
        self.n_seen += chunk.shape[1]
        outs = []
        while self.n_seen >= self.next_end:
            outs.append(self._process(self.next_end))
            self.next_end += self.step
        keep_from = max(0, self.next_end - self.win - self.context) - self.buffer_offset
        if keep_from > 0:                                            # drop samples no longer needed
            self.buffer = self.buffer[:, keep_from:]
            self.buffer_offset += keep_from
        return outs

    def run_recording(self, data: np.ndarray) -> list[CognitiveStateOutput]:
        """Process a whole recording (channels x samples) as if it streamed in."""
        self.reset()
        return self.push(data)

    # ------------------------------------------------------------------ core
    def _process(self, end: int) -> CognitiveStateOutput:
        t0 = time.perf_counter()
        s = max(0, end - self.context - self.win) - self.buffer_offset
        e = end - self.buffer_offset
        filtered = filter_recording(self.buffer[:, s:e], self.fs, self.pcfg["filter"])
        window = filtered[:, -self.win:]

        # quality + persistent bad channels (rolling history of hard failures)
        q = assess_window(window, self.fs, self.qcfg, self.ch)
        self.history.append(q["channel_bad"] & ~q["blink"])
        interp = self._persistent_bad()
        if interp:
            window = interpolate_channels(window, self.fs, self.ch, interp)
            q = assess_window(window, self.fs, self.qcfg, self.ch)
        bad_names = [c for c, b in zip(self.ch, q["channel_bad"]) if b]

        # features (identical code path to training)
        F, arrays = extract_features(window[None], self.fs, self.ch, self.fcfg)
        bp = arrays["bandpower"][0]
        b_idx = self.conn_bands.index(self.band)
        entry = {"epoch": window.astype(np.float32), "features": F, "bandpower": bp,
                 "adj": arrays["plv"][0].mean(axis=0), "plv_band": arrays["plv"][0, b_idx]}
        self.seq.append(entry)

        p = np.asarray(self.predictor.predict(list(self.seq)), dtype=float)
        p = p / p.sum()
        h = heuristic_indices(bp, self.ch)
        self.ema_p = p if self.ema_p is None else self.alpha * p + (1 - self.alpha) * self.ema_p
        hv = np.array([h["attention"], h["fatigue"], h["readiness"]])
        self.ema_h = hv if self.ema_h is None else self.alpha * hv + (1 - self.alpha) * self.ema_h

        classes = self.predictor.classes
        t_start_s, t_end_s = (end - self.win) / self.fs, end / self.fs
        out = CognitiveStateOutput(
            participant_id=self.participant_id,
            session_id=self.session_id,
            window_index=self.window_index,
            timestamp_start=(self.session_start + dt.timedelta(seconds=t_start_s)).isoformat(),
            timestamp_end=(self.session_start + dt.timedelta(seconds=t_end_s)).isoformat(),
            t_start_s=t_start_s, t_end_s=t_end_s,
            signal_quality=SignalQuality(overall=float(q["score"]), flag=q["flag"], bad_channels=bad_names,
                                         interpolated_channels=interp),
            states=States(
                workload=WorkloadState(
                    label=classes[int(np.argmax(self.ema_p))],
                    probabilities={c: float(v) for c, v in zip(classes, self.ema_p)},
                    raw_probabilities={c: float(v) for c, v in zip(classes, p)},
                    confidence=float(self.ema_p.max()), model_version=self.predictor.version,
                    task=self.predictor.task),
                attention=HeuristicState(value=float(self.ema_h[0]), raw_value=h["attention"], formula=FORMULAS["attention"]),
                fatigue=HeuristicState(value=float(self.ema_h[1]), raw_value=h["fatigue"], formula=FORMULAS["fatigue"]),
                readiness=HeuristicState(value=float(self.ema_h[2]), raw_value=h["readiness"], formula=FORMULAS["readiness"]),
            ),
            band_power={c: {b: float(bp[k, j]) for j, b in enumerate(BANDS)} for k, c in enumerate(self.ch)},
            connectivity=Connectivity(metric="plv", band=self.band, channels=self.ch,
                                      matrix=np.round(entry["plv_band"], 4).tolist()),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self.last_window = window
        self.window_index += 1
        return out

    def _persistent_bad(self) -> list[str]:
        cfg = self.pcfg["bad_channels"]
        if not cfg.get("interpolate", True) or len(self.history) < cfg.get("min_history_windows", 10):
            return []
        frac = np.mean(np.stack(self.history), axis=0)
        cand = [(frac[k], c) for k, c in enumerate(self.ch) if frac[k] >= cfg.get("min_bad_fraction", 0.6)]
        cand.sort(reverse=True)
        return [c for _, c in cand[: cfg.get("max_interpolated", 3)]]
