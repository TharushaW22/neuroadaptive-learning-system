"""Vision adapter for Component 2's PyTorch GRU (DAiSEE engagement, binary: low = 0-1, high = 2-3).

Instance = one window of C2's TEST split: 20 frames (4 s at 5 fps, stride 2 s) x 15 per-frame facial
features, standardised with C2's training-split normaliser (data/vision_data.py). The explained output
is P(target class), by default P(high engagement).

Interpretable space (representation/vision_space.py): 15 features x S time segments (default 4 x 1 s),
grouped into face regions (eyes/blink, gaze, head pose, mouth, brows) in the context YAML.

  "absent"   off-manifold = training mean (0 after standardisation)       -> fidelity_off, IG zero baseline
             on-manifold  = frames of donor windows of TRAINING-split subjects -> KernelSHAP, LIME, IG donor
             conditional  = Gaussian conditional on segment means (Ledoit-Wolf), or the empirical
                            nearest-donor conditional when the Gaussian check fails (replacement/vision_strategies.py)
  in-sample  test windows are OUT-OF-SAMPLE only when C2's split is verified subject-disjoint; otherwise
             every output is labelled IN-SAMPLE.

C2 has one official train/validation/test split (no folds): every instance has fold 0 and the
"training windows of fold 0" are the train-split windows.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from xaicomp.adapters.base import Capabilities, InstanceView, ModelAdapter
from xaicomp.data.vision_data import VisionData, c2_paths, load_vision_data
from xaicomp.models.vision_c2 import (C2_FEATURE_COLUMNS, LoadedVisionModel, VisionModelError, load_vision_model,
                                      randomise_vision, select_model_dir)
from xaicomp.replacement.strategies import nearest_donors
from xaicomp.replacement.vision_strategies import (GaussianVector, compose_frames, conditional_shifts,
                                                   gaussian_suitability, segment_means)
from xaicomp.representation.vision_space import VisionSpace, segment_names


# context-specific wording for the screening table (screening/applicability.py)
VISION_SCREENING_NOTES = {
    "not_actionable": "per-frame facial measurements are coupled (e.g. left/right/mean eye aspect ratio) and are "
                      "not independently actionable inputs a learner could change",
    "binning": "rules need discretised features; the 60 feature x segment values must first be binned"}


def stable_int(text: str) -> int:
    return zlib.crc32(text.encode())


class VisionGRUModule(nn.Module):
    """Differentiable path for Integrated Gradients: frames (B, F, S, L) + mask (B, T) -> P(target class).

    Summing IG over the L frames of a segment gives one attribution per feature x segment, and
    completeness carries over (the module only re-arranges the input)."""

    def __init__(self, net: nn.Module, target_idx: int):
        super().__init__()
        self.net = net.eval()
        self.target_idx = target_idx

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, F, S, L = frames.shape
        x = frames.reshape(B, F, S * L).transpose(1, 2)                  # (B, T, F)
        return torch.softmax(self.net(x, mask), dim=-1)[:, self.target_idx]


class VisionBank:
    """Mirrors the EEG ModelBank interface: one model for every instance (C2 has no folds)."""

    def __init__(self, lm: LoadedVisionModel, in_sample: bool):
        self.lm = lm
        self.source = "c2_trained"
        self._in_sample = in_sample

    @property
    def in_sample(self) -> bool:
        return self._in_sample

    def get(self, fold: int = 0) -> LoadedVisionModel:
        return self.lm


class VisionComponent2Adapter(ModelAdapter):
    modality = "vision"

    def __init__(self, ctx: dict, data: VisionData | None = None, model: LoadedVisionModel | None = None,
                 threads: int = 4, batch: int = 512):
        self.ctx = ctx
        self.context_id = ctx["context_id"]
        torch.set_num_threads(threads)
        if model is None:
            d, report = select_model_dir(c2_paths(ctx)["models_dir"], ctx["component2"].get("model_name", "auto"),
                                         ctx["expect"])
            if d is None:
                raise VisionModelError(f"no usable Component 2 model: {report or 'models folder empty'} — "
                                       "run scripts/check_vision_ready.py")
            model = load_vision_model(d)
        self.trained = model
        lm = randomise_vision(model, int(ctx.get("randomisation_seed", 1234))) if ctx.get("weights") == "randomised" else model
        self.data = data if data is not None else load_vision_data(ctx, model, use_cache=ctx.get("cache_windows", True))
        self.batch = batch
        names = ctx.get("class_labels") or lm.class_names
        if len(names) != len(lm.class_names):
            raise ValueError("class_labels must have one entry per model class")
        self.class_names = list(names)
        self.model_class_names = list(lm.class_names)
        self.target_class = ctx.get("target_class", self.class_names[-1])
        self.target_idx = self.class_names.index(self.target_class)

        ws = self.data.windows
        self.T, self.F = ws.X.shape[1], ws.X.shape[2]
        rep = ctx["representation"]
        self.S = int(rep["segments"])
        if self.T % self.S:
            raise ValueError(f"{self.T} frames cannot be cut into {self.S} equal segments")
        self.L = self.T // self.S
        fps = float(lm.meta.get("sampling_fps", 5))
        self.space = VisionSpace(list(C2_FEATURE_COLUMNS), segment_names(self.S, self.T, fps), ctx["groups"])
        self.bank = VisionBank(lm, in_sample=not self.data.split.get("subject_disjoint", False))
        full = ws.mask.min(axis=1) > 0.5
        self._train_pos = np.where((ws.splits == "train") & full)[0]
        self._windows = self._build_windows(fps)
        self._z = None
        self._gauss = None
        self._cond = None
        self._pool = None
        self._noise_exclude = [C2_FEATURE_COLUMNS.index(f) for f in ctx.get("robustness_exclude", [])]

    # ------------------------------------------------------------------ instances
    def _build_windows(self, fps: float) -> pd.DataFrame:
        ws = self.data.windows
        pos = np.where(ws.splits == "test")[0]
        pids = pd.Series(ws.participant_ids[pos])
        # subject_id must be an int everywhere (statistics, export); DAiSEE IDs are digit strings
        codes = {p: (int(p) if p.isdigit() else 900000 + i) for i, p in enumerate(sorted(pids.unique()))}
        rows = []
        for p in pos:
            vid, start = str(ws.video_ids[p]), int(ws.start_frame[p])
            y = int(self.data.y[p])
            rows.append({"instance_id": f"v{vid}_f{start:03d}", "subject_id": codes[str(ws.participant_ids[p])],
                         "participant_id": str(ws.participant_ids[p]), "video_id": vid,
                         "condition": self.class_names[y], "y": y, "label_level": int(ws.y_level[p]), "fold": 0,
                         "split": "test", "start_frame": start, "t_start": float(ws.start_ts[p]),
                         "t_end": float(ws.start_ts[p]) + self.T / fps, "frac_bad": float(1.0 - ws.quality_share[p]),
                         "pos": int(p)})
        df = pd.DataFrame(rows)
        if len(df) and df["instance_id"].duplicated().any():
            raise ValueError("duplicate window ids")
        return df.set_index("instance_id", drop=False) if len(df) else df

    def instances(self) -> pd.DataFrame:
        return self._windows.drop(columns=["pos"])

    @property
    def in_sample(self) -> bool:
        return self.bank.in_sample

    @property
    def in_sample_label(self) -> str:
        return ("IN-SAMPLE — Component 2's split is not verified subject-disjoint; not for final results"
                if self.in_sample else "")

    def capabilities(self) -> Capabilities:
        return Capabilities(differentiable=True, n_features=self.space.n_features, model_kind="deep (GRU)",
                            supports_conditional=True,
                            notes={"model_source": self.bank.source, "weights": self.ctx.get("weights", "trained"),
                                   "segments": self.S, "frames_per_segment": self.L, **VISION_SCREENING_NOTES})

    # ------------------------------------------------------------------ frames <-> model input
    def frames_at(self, positions) -> np.ndarray:
        """(n, F, S, L) frames of windows at global positions."""
        X = self.data.X[np.asarray(positions)]                           # (n, T, F)
        return np.ascontiguousarray(X.transpose(0, 2, 1)).reshape(len(X), self.F, self.S, self.L)

    def to_model(self, frames: np.ndarray) -> np.ndarray:
        n = len(frames)
        return np.ascontiguousarray(frames.reshape(n, self.F, self.T).transpose(0, 2, 1))

    def predict_frames(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Class probabilities (n, C) for frames (n, F, S, L) with one mask (T,) or masks (n, T)."""
        lm = self.bank.get()
        mask = np.broadcast_to(mask, (len(frames), self.T)) if mask.ndim == 1 else mask
        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), self.batch):
                x = torch.from_numpy(self.to_model(frames[i:i + self.batch]))
                m = torch.from_numpy(np.array(mask[i:i + self.batch], dtype=np.float32))
                out.append(torch.softmax(lm.model(x, m).double(), dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.class_names)))

    def predict_instances(self, instance_ids: list[str]) -> np.ndarray:
        pos = self._windows.loc[list(instance_ids), "pos"].to_numpy()
        return self.predict_frames(self.frames_at(pos), self.data.windows.mask[pos])

    def torch_module(self) -> VisionGRUModule:
        return VisionGRUModule(self.bank.get().model, self.target_idx)

    # ------------------------------------------------------------------ backgrounds
    def training_windows(self, fold: int = 0) -> np.ndarray:
        """Positions of full-length TRAIN-split windows (training subjects only)."""
        return self._train_pos

    def zmeans(self) -> np.ndarray:
        """(N, F*S) segment means of every window (the vector the Gaussian conditional is fitted on)."""
        if self._z is None:
            self._z = segment_means(self.frames_at(np.arange(len(self.data.X))))
        return self._z

    def gaussian(self, fold: int = 0) -> GaussianVector:
        if self._gauss is None:
            self._gauss = GaussianVector(self.zmeans()[self._train_pos])
        return self._gauss

    def conditional_check(self) -> dict:
        """Gaussian vs empirical conditional for Conditional SHAP (context YAML `conditional`)."""
        if self._cond is None:
            cfg = self.ctx.get("conditional", {})
            mode = cfg.get("mode", "auto")
            if mode in ("gaussian", "empirical"):
                self._cond = {"mode": mode, "reason": "fixed in the context YAML"}
            else:
                tp = self._train_pos
                self._cond = gaussian_suitability(self.zmeans()[tp], self.data.windows.participant_ids[tp],
                                                  nominal=float(cfg.get("nominal", 0.90)),
                                                  coverage_range=tuple(cfg.get("coverage_range", (0.85, 0.95))),
                                                  max_discrete_share=float(cfg.get("max_discrete_share", 0.10)))
        return self._cond

    @property
    def conditional_mode(self) -> str:
        return self.conditional_check()["mode"]

    def donor_positions(self, iid: str, k: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng([seed, stable_int(iid), 17])
        return rng.choice(self._train_pos, size=k, replace=False)

    def empirical_pool(self, size: int = 128, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if self._pool is None:
            rng = np.random.default_rng([seed, 0, 99])
            pos = rng.choice(self._train_pos, size=min(size, len(self._train_pos)), replace=False)
            self._pool = (pos, self.zmeans()[pos])
        return self._pool

    def local_feature_correlation(self, iid: str) -> float:
        """Feature dependence inside the window: mean |Pearson r| between the 15 features across its valid
        frames (stored in the same column as EEG's band-power correlation, `local_bp_corr`)."""
        p = int(self._windows.loc[iid, "pos"])
        X = self.data.X[p][self.data.windows.mask[p] > 0.5].T              # (F, T_valid)
        X = X[X.std(axis=1) > 1e-9]
        if len(X) < 2:
            return float("nan")
        R = np.corrcoef(X)
        iu = np.triu_indices_from(R, k=1)
        return float(np.nanmean(np.abs(R[iu])))

    def model_identity(self) -> str:
        """Which C2 checkpoint (name + creation time): a run is never resumed with a different one."""
        m = self.trained.meta
        return f"{m.get('model_name')}@{m.get('created_utc')}"

    def run_info_extra(self) -> dict:
        lm = self.bank.get()
        return {"modality": "vision", "label": self.in_sample_label, "model_class_names": self.model_class_names,
                "c2_model": {"name": lm.meta.get("model_name"), "path": str(lm.path), "created_utc": lm.meta.get("created_utc"),
                             "label_scheme": lm.meta["label_scheme"].get("name"), "cell": lm.meta["model_params"].get("cell"),
                             "library_versions": lm.meta.get("library_versions")},
                "data_source": self.data.source, "split_check": self.data.split,
                "conditional_check": self.conditional_check()}

    def view(self, instance_id: str, k_donors: int, seed: int, noise_seed: int | None = None,
             noise_scale: float = 0.1) -> "VisionInstanceView":
        """Instance view; with `noise_seed`, Gaussian noise (sigma = noise_scale training SDs, i.e. noise_scale in
        standardised units) is added to the valid frames (robustness), except for `robustness_exclude` features."""
        noise = None
        if noise_seed is not None:
            rng = np.random.default_rng([noise_seed, 51])
            noise = (rng.standard_normal((self.F, self.S, self.L)) * noise_scale).astype(np.float32)
            noise[self._noise_exclude] = 0.0
        return VisionInstanceView(self, instance_id, k_donors, seed, noise)


class VisionInstanceView(InstanceView):
    """One window ready for explanation: frames (F, S, L), mask, donors and the evaluation counter."""

    def __init__(self, ad: VisionComponent2Adapter, iid: str, k_donors: int, seed: int, noise: np.ndarray | None = None):
        self.ad, self.instance_id, self.seed = ad, iid, seed
        w = ad._windows.loc[iid]
        self.fold, self.subject_id, self.pos = 0, int(w["subject_id"]), int(w["pos"])
        self.mask = ad.data.windows.mask[self.pos].astype(np.float32)
        self.x = ad.frames_at([self.pos])[0]
        if noise is not None:
            valid = self.mask.reshape(ad.S, ad.L)[None]                    # noise only on real frames
            self.x = (self.x + noise * valid).astype(np.float32)
        self.n_features = ad.space.n_features
        self.z_x = segment_means(self.x)
        self.k_donors = k_donors
        self.donor_pos = ad.donor_positions(iid, k_donors, seed) if k_donors > 0 else np.zeros(0, int)
        self.donors = ad.frames_at(self.donor_pos) if k_donors > 0 else np.zeros((0, ad.F, ad.S, ad.L), np.float32)
        self.donor_z = ad.zmeans()[self.donor_pos] if k_donors > 0 else np.zeros((0, self.n_features))
        self.conditional_mode = ad.conditional_mode
        self.n_evals = 0
        self._fx = None
        self.last_conditional_targets = None

    def _p(self, frames: np.ndarray) -> np.ndarray:
        self.n_evals += len(frames)
        return self.ad.predict_frames(frames, self.mask)[:, self.ad.target_idx]

    def fx(self) -> float:
        if self._fx is None:
            self._fx = float(self._p(self.x[None])[0])
        return self._fx

    def value(self, masks: np.ndarray, replacement: str, rng: np.random.Generator | None = None,
              return_draws: bool = False) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, dtype=bool))
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n, (F, S) = len(masks), (self.ad.F, self.ad.S)
        if replacement == "zero":
            return self._p(compose_frames(self.x, masks, None))
        K = self.k_donors
        if K == 0:
            raise ValueError("this view has no donors (k_donors=0)")
        out = np.zeros((n, K))
        if replacement == "donor":
            for k in range(K):
                out[:, k] = self._p(compose_frames(self.x, masks, self.donors[k]))
        elif replacement == "conditional":
            shifts, targets = conditional_shifts(masks, self.z_x, self.donor_z, self.ad.gaussian(), rng)
            self.last_conditional_targets = targets
            for k in range(K):
                out[:, k] = self._p(compose_frames(self.x, masks, self.donors[k], shifts[:, k].reshape(n, F, S)))
        elif replacement == "empirical":
            pos, pool_z = self.ad.empirical_pool()
            scale = pool_z.std(axis=0) + 1e-9
            for i in range(n):
                near = nearest_donors(masks[i], self.z_x, pool_z, scale, K, rng)
                donors = self.ad.frames_at(pos[near])
                out[i] = self._p(np.concatenate([compose_frames(self.x, masks[i:i + 1], d) for d in donors]))
        else:
            raise ValueError(f"unknown replacement {replacement!r}")
        return out if return_draws else out.mean(axis=1)

    def torch_parts(self):
        """(differentiable module, input tensor (1, F, S, L), extra forward args (mask,)) for gradient methods."""
        return self.ad.torch_module(), torch.as_tensor(self.x[None]), (torch.as_tensor(self.mask[None]),)

    def mean_donor_value(self) -> float:
        return float(self.value(np.zeros((1, self.n_features), bool), "donor")[0])
