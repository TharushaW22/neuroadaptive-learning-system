"""EEG adapter for Component 1's deep sequence models (primary: T1 A8 CNN-LSTM-Transformer).

Instance = one 18-s window = 8 consecutive 4-s epochs of one recording (C1's model input), on the
same grid C1 uses for its test sequences (start every 2 epochs). The window is rebuilt from the
CLEANED CONTINUOUS recording (14 x 2304 samples) and split into 70 channel x band components +
a residual (representation/eeg_bands.py). Model input is formed exactly as in C1:

  epochs (8, 14, 512) = 50 %-overlapping slices of the window (step 256 samples = 2 s)
  x = clip(epochs / channel_scale, -clip, +clip)              (trainer.py:80-81, predictors.py:60-61)
  p = softmax(model(x)["workload"] / temperature)             (predictors.py:68-70)

The target output explained is P(target class), by default P("high" workload).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn

from xaicomp import _bootstrap  # noqa: F401
from xaicomp.adapters.base import Capabilities, InstanceView, ModelAdapter
from xaicomp.config import load_c3_config
from xaicomp.data.eeg_data import EEGData, load_data, processed_dir
from xaicomp.models.loader import LoadedModel, ModelBank
from xaicomp.paths import C1_ROOT, safe_path
from xaicomp.replacement.strategies import (GaussianLogPower, compose, conditional_gains,
                                            nearest_donors)
from xaicomp.representation.eeg_bands import decompose, log_power
from xaicomp.representation.groups import InterpretableSpace

EPOCH_LEN = 512          # 4 s x 128 Hz (C1 configs/preprocess.yaml epoching.window_s)
EPOCH_STEP = 256         # 2 s x 128 Hz (epoching.step_s)


def stable_int(text: str) -> int:
    return zlib.crc32(text.encode())


class BandComposedModel(nn.Module):
    """Differentiable path: band components (B, C, bands, T) + residual -> P(target class).

    Used by Integrated Gradients. Gradients through the clip are zero where |x| > clip, exactly
    as in the deployed model."""

    def __init__(self, lm: LoadedModel, seq_len: int, target_idx: int):
        super().__init__()
        self.net = lm.model.eval()
        self.register_buffer("scale", torch.as_tensor(lm.channel_scale, dtype=torch.float32))
        self.clip, self.T, self.L, self.target_idx = float(lm.clip), float(lm.temperature), seq_len, target_idx

    def forward(self, comps: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = comps.sum(dim=2) + residual                                  # (B, C, T)
        ep = x.unfold(2, EPOCH_LEN, EPOCH_STEP)[:, :, : self.L]          # (B, C, L, 512)
        ep = ep.permute(0, 2, 1, 3) / self.scale.view(1, 1, -1, 1)
        ep = torch.clamp(ep, -self.clip, self.clip)
        logits = self.net(ep)["workload"] / self.T
        return torch.softmax(logits, dim=-1)[:, self.target_idx]


def segments_to_epochs(seg: np.ndarray, seq_len: int) -> np.ndarray:
    """(n, C, T) window -> (n, L, C, 512) epochs, as C1 slices them."""
    return np.stack([seg[..., j * EPOCH_STEP: j * EPOCH_STEP + EPOCH_LEN] for j in range(seq_len)], axis=1)


class EEGComponent1Adapter(ModelAdapter):
    modality = "eeg"

    def __init__(self, ctx: dict, data: EEGData | None = None, bank: ModelBank | None = None,
                 threads: int = 4, cache_recordings: int = 12):
        from eegengine.io.stew import CHANNELS
        from eegengine.labels import TASKS
        from eegengine.training.cv import load_assignment

        self.ctx = ctx
        self.context_id = ctx["context_id"]
        self.data = data if data is not None else load_data(epochs=False)
        self.bank = bank if bank is not None else ModelBank(ctx)
        self.task = ctx["task"]
        self.class_names = list(TASKS[self.task]["classes"])
        self.target_class = ctx.get("target_class", self.class_names[-1])
        self.target_idx = self.class_names.index(self.target_class)
        rep = ctx["representation"]
        self.bands = {k: tuple(v) for k, v in rep["bands"].items()}
        self.crossover = float(rep.get("crossover_hz", 1.0))
        self.space = InterpretableSpace(list(CHANNELS), list(self.bands), ctx.get("regions", {}))
        self.L = int(ctx["window"]["seq_len"])
        self.stride = int(ctx["window"]["stride_epochs"])
        self.seg_len = (self.L - 1) * EPOCH_STEP + EPOCH_LEN             # 2304 samples = 18 s
        self.assignment = load_assignment(C1_ROOT / load_c3_config("data.yaml")["c1_folds_file"])
        self._windows = self._build_windows()
        self._rec_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_n = cache_recordings
        self._z = None
        self._gauss: dict[int, GaussianLogPower] = {}
        self._pool: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        torch.set_num_threads(threads)

    # ------------------------------------------------------------------ instances
    def _build_windows(self) -> pd.DataFrame:
        from eegengine.labels import TASKS

        meta = self.data.meta
        col = TASKS[self.task]["column"]
        rows = []
        for rec, g in meta[meta[col] >= 0].groupby("rec_index", sort=True):
            g = g.sort_values("epoch_index")
            idx = g.index.to_numpy()
            for start in range(0, len(idx) - self.L + 1, self.stride):
                ep = idx[start:start + self.L]
                first = meta.loc[ep[0]]
                s = int(first["subject_id"])
                rows.append({
                    "instance_id": f"s{s:02d}_{first['condition']}_e{start:02d}",
                    "subject_id": s, "condition": first["condition"], "rec_index": int(rec),
                    "start_epoch": int(start), "start_sample": int(start * EPOCH_STEP),
                    "t_start": float(first["t_start"]), "t_end": float(meta.loc[ep[-1], "t_end"]),
                    "y": int(first[col]), "fold": int(self.assignment[s]),
                    "frac_bad": float((meta.loc[ep, "quality_flag"] == "bad").mean()),
                    "epoch_rows": ep.tolist(),
                })
        return pd.DataFrame(rows).set_index("instance_id", drop=False)

    def instances(self) -> pd.DataFrame:
        return self._windows.drop(columns=["epoch_rows"])

    @property
    def in_sample(self) -> bool:
        return self.bank.in_sample

    def capabilities(self) -> Capabilities:
        return Capabilities(differentiable=True, n_features=self.space.n_features, model_kind="deep",
                            notes={"model_source": self.bank.source, "weights": self.ctx.get("weights", "trained"),
                                   "seg_len": self.seg_len, "seq_len": self.L})

    # ------------------------------------------------------------------ signals
    def recording_components(self, rec: int) -> tuple[np.ndarray, np.ndarray]:
        """(components (C, B, N) float32, residual (C, N) float32) of one whole recording (cached)."""
        if rec in self._rec_cache:
            self._rec_cache.move_to_end(rec)
            return self._rec_cache[rec]
        comps, resid = decompose(self.data.continuous[rec], self.data.fs, self.bands, self.crossover)
        out = (comps.astype(np.float32), resid.astype(np.float32))
        self._rec_cache[rec] = out
        if len(self._rec_cache) > self._cache_n:
            self._rec_cache.popitem(last=False)
        return out

    def window_signal(self, iid: str) -> np.ndarray:
        w = self._windows.loc[iid]
        return self.data.continuous[w["rec_index"], :, w["start_sample"]: w["start_sample"] + self.seg_len]

    def window_components(self, iid: str) -> tuple[np.ndarray, np.ndarray]:
        w = self._windows.loc[iid]
        comps, resid = self.recording_components(int(w["rec_index"]))
        s = int(w["start_sample"])
        return comps[..., s:s + self.seg_len], resid[:, s:s + self.seg_len]

    def logpower_table(self) -> np.ndarray:
        """(n_windows, 70) log10 band power of every window's components; cached on disk."""
        if self._z is not None:
            return self._z
        key = hashlib.sha1(json.dumps([self.bands, self.crossover, self.L, self.stride, self.task]).encode()).hexdigest()[:8]
        path = processed_dir() / f"window_logpower_{key}.npy"
        if path.exists():
            self._z = np.load(path)
        else:
            Z = np.empty((len(self._windows), self.space.n_features))
            for rec, g in self._windows.groupby("rec_index"):
                comps, _ = self.recording_components(int(rec))
                for pos, s in zip(self._windows.index.get_indexer(g.index), g["start_sample"]):
                    Z[pos] = log_power(comps[..., s:s + self.seg_len]).reshape(-1)
            np.save(safe_path(path), Z)
            self._z = Z
        return self._z

    # ------------------------------------------------------------------ model
    def model_for(self, fold: int) -> LoadedModel:
        return self.bank.get(fold)

    def predict_segments(self, seg: np.ndarray, fold: int, batch: int = 16) -> np.ndarray:
        """Calibrated class probabilities (n, n_classes) for windows (n, C, T) in µV.

        Batch 16 is deliberate: measured on the 8.6 GB dev Mac, 8-16 windows cost ~17 ms each with
        ~0.4 GB peak memory, 64 windows ~31 ms, 128 windows ~227 ms (memory thrashing)."""
        lm = self.model_for(fold)
        out = []
        with torch.inference_mode():
            for i in range(0, len(seg), batch):
                ep = segments_to_epochs(seg[i:i + batch], self.L) / lm.channel_scale[None, None, :, None]
                ep = np.clip(ep, -lm.clip, lm.clip).astype(np.float32)
                logits = lm.model(torch.from_numpy(ep))["workload"] / lm.temperature
                out.append(torch.softmax(logits.double(), dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.class_names)))

    def p_target(self, seg: np.ndarray, fold: int) -> np.ndarray:
        return self.predict_segments(seg, fold)[:, self.target_idx]

    def predict_instances(self, instance_ids: list[str]) -> np.ndarray:
        out = np.zeros((len(instance_ids), len(self.class_names)))
        w = self._windows.loc[list(instance_ids)]
        for fold, g in w.groupby("fold"):
            seg = np.stack([self.window_signal(i) for i in g.index])
            out[w.index.get_indexer(g.index)] = self.predict_segments(seg, int(fold))
        return out

    def torch_module(self, fold: int) -> BandComposedModel:
        return BandComposedModel(self.model_for(fold), self.L, self.target_idx)

    # ------------------------------------------------------------------ backgrounds
    def training_windows(self, fold: int) -> np.ndarray:
        """Positions of windows whose subject is in the TRAINING folds of `fold` (never the test subjects)."""
        return np.where(self._windows["fold"].to_numpy() != fold)[0]

    def gaussian(self, fold: int) -> GaussianLogPower:
        if fold not in self._gauss:
            self._gauss[fold] = GaussianLogPower(self.logpower_table()[self.training_windows(fold)])
        return self._gauss[fold]

    def donor_positions(self, iid: str, k: int, seed: int) -> np.ndarray:
        fold = int(self._windows.loc[iid, "fold"])
        rng = np.random.default_rng([seed, stable_int(iid), 17])
        return rng.choice(self.training_windows(fold), size=k, replace=False)

    def components_at(self, positions: np.ndarray) -> np.ndarray:
        return np.stack([self.window_components(self._windows.index[p])[0] for p in positions])

    def empirical_pool(self, fold: int, size: int = 128, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Pool of training windows for the empirical conditional: (positions, log powers)."""
        if fold not in self._pool:
            rng = np.random.default_rng([seed, fold, 99])
            pos = rng.choice(self.training_windows(fold), size=min(size, len(self.training_windows(fold))), replace=False)
            self._pool[fold] = (pos, self.logpower_table()[pos])
        return self._pool[fold]

    def view(self, instance_id: str, k_donors: int, seed: int, noise_seed: int | None = None,
             noise_scale: float = 0.1) -> "EEGInstanceView":
        """Instance view; with `noise_seed`, Gaussian noise (sigma = noise_scale x channel scale of the
        instance's model) is added to the window (robustness metric)."""
        noise = None
        if noise_seed is not None:
            from xaicomp.metrics.robustness import noise_for_window
            fold = int(self._windows.loc[instance_id, "fold"])
            noise = noise_for_window(self.model_for(fold).channel_scale, (len(self.space.channels), self.seg_len),
                                     noise_scale, noise_seed)
        return EEGInstanceView(self, instance_id, k_donors, seed, noise)


class EEGInstanceView(InstanceView):
    """One window ready for explanation: components, residual, donors and the evaluation counter."""

    def __init__(self, ad: EEGComponent1Adapter, iid: str, k_donors: int, seed: int, noise: np.ndarray | None = None):
        self.ad, self.instance_id, self.seed = ad, iid, seed
        w = ad._windows.loc[iid]
        self.fold, self.subject_id = int(w["fold"]), int(w["subject_id"])
        self.x, self.residual = ad.window_components(iid)
        if noise is not None:          # noise is split by the same filter bank, so signal + noise is exact
            nc, nr = decompose(noise, ad.data.fs, ad.bands, ad.crossover, pad=min(512, noise.shape[-1] - 1))
            self.x = (self.x + nc).astype(np.float32)
            self.residual = (self.residual + nr).astype(np.float32)
        self.C, self.B, self.T = self.x.shape
        self.n_features = self.C * self.B
        self.z_x = log_power(self.x).reshape(-1)
        self.k_donors = k_donors
        self.donor_pos = ad.donor_positions(iid, k_donors, seed) if k_donors > 0 else np.zeros(0, int)
        self.donors = ad.components_at(self.donor_pos) if k_donors > 0 else np.zeros((0, self.C, self.B, self.T), np.float32)
        self.donor_z = ad.logpower_table()[self.donor_pos] if k_donors > 0 else np.zeros((0, self.n_features))
        self.n_evals = 0
        self._fx = None
        self.last_conditional_targets = None

    def _p(self, seg: np.ndarray) -> np.ndarray:
        self.n_evals += len(seg)
        return self.ad.p_target(seg, self.fold)

    def fx(self) -> float:
        if self._fx is None:
            self._fx = float(self._p(compose(self.x, self.residual, np.ones((1, self.n_features)), None))[0])
        return self._fx

    def segment(self) -> np.ndarray:
        return compose(self.x, self.residual, np.ones((1, self.n_features)), None)[0]

    def value(self, masks: np.ndarray, replacement: str, rng: np.random.Generator | None = None,
              chunk: int = 16, return_draws: bool = False) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, dtype=bool))
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = len(masks)
        if replacement == "zero":
            return np.concatenate([self._p(compose(self.x, self.residual, masks[i:i + chunk], None))
                                   for i in range(0, n, chunk)])
        K = self.k_donors
        if K == 0:
            raise ValueError("this view has no donors (k_donors=0)")
        out = np.zeros((n, K))
        if replacement == "donor":
            for k in range(K):
                for i in range(0, n, chunk):
                    out[i:i + chunk, k] = self._p(compose(self.x, self.residual, masks[i:i + chunk], self.donors[k]))
        elif replacement == "conditional":
            gains, targets = conditional_gains(masks, self.z_x, self.donor_z, self.ad.gaussian(self.fold), rng)
            self.last_conditional_targets = targets
            for k in range(K):
                for i in range(0, n, chunk):
                    g = gains[i:i + chunk, k].reshape(-1, self.C, self.B)
                    out[i:i + chunk, k] = self._p(compose(self.x, self.residual, masks[i:i + chunk], self.donors[k], g))
        elif replacement == "empirical":
            pos, pool_z = self.ad.empirical_pool(self.fold)
            scale = pool_z.std(axis=0) + 1e-9
            for i in range(n):
                near = nearest_donors(masks[i], self.z_x, pool_z, scale, K, rng)
                donors = self.ad.components_at(pos[near])
                segs = np.concatenate([compose(self.x, self.residual, masks[i:i + 1], d) for d in donors])
                out[i] = self._p(segs)
        else:
            raise ValueError(f"unknown replacement {replacement!r}")
        return out if return_draws else out.mean(axis=1)

    def torch_parts(self):
        """(differentiable module, input tensor (1, C, B, T), extra forward args) for gradient methods."""
        module = self.ad.torch_module(self.fold)
        return module, torch.as_tensor(self.x[None]), (torch.as_tensor(self.residual[None]),)

    def mean_donor_value(self) -> float:
        """Model output with EVERY component taken from the donor (= SHAP's v(empty), IG-donor's baseline)."""
        return float(self.value(np.zeros((1, self.n_features), bool), "donor")[0])
