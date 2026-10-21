"""Fidelity: does removing the features an explanation calls important really change the prediction?

For one instance and one attribution vector a:
  * s = a pointed at the PREDICTED class (a if the prediction is the target class, else -a)
  * order features from most to least supportive of the prediction (MoRF)
  * deletion curve:  remove the first k features (k = 0, step, 2*step, ..., F), track P(predicted class)
                     -> a faithful explanation makes it DROP fast   (low area = good)
  * insertion curve: start with every feature removed and add the first k back
                     -> a faithful explanation makes it RISE fast   (high area = good)
Areas are normalised to x in [0, 1]. Reference orders on the same instance:
  * random orders (n_random, the same for every method)  -> what "no information" scores
  * reverse order (least relevant first, LeRF)            -> the worst case for this attribution
Score (higher = better):  fidelity = 0.5 * [(AUC_del_random - AUC_del) + (AUC_ins - AUC_ins_random)]
i.e. how much better than random the explanation orders the features, in probability units.

Two removal modes (D: off- vs on-manifold):
  off_manifold  removed components are set to zero
  on_manifold   removed components are replaced by donor components (K donors of training subjects,
                drawn with a FIXED evaluation seed, identical for all methods)
"""

from __future__ import annotations

import numpy as np


def ks(F: int, step: int) -> np.ndarray:
    k = list(range(0, F + 1, step))
    if k[-1] != F:
        k.append(F)
    return np.array(k)


def curve_masks(order: np.ndarray, F: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(deletion masks, insertion masks), each (n_points, F) of PRESENT features."""
    kk = ks(F, step)
    dele = np.ones((len(kk), F), dtype=bool)
    ins = np.zeros((len(kk), F), dtype=bool)
    for i, k in enumerate(kk):
        dele[i, order[:k]] = False
        ins[i, order[:k]] = True
    return dele, ins


def auc(curve: np.ndarray) -> float:
    x = np.linspace(0, 1, len(curve))
    return float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(x)))


def to_pred(p_target: np.ndarray, pred_is_target: bool) -> np.ndarray:
    return p_target if pred_is_target else 1.0 - p_target


def evaluate_orders(view, orders: dict[str, np.ndarray], step: int, replacement: str, pred_is_target: bool,
                    rng_seed: int = 0) -> dict[str, dict]:
    """Deletion/insertion curves for several feature orders in ONE batched model call."""
    F = view.n_features
    blocks, names = [], []
    for name, order in orders.items():
        d, i = curve_masks(order, F, step)
        blocks += [d, i]
        names.append(name)
    M = np.concatenate(blocks)
    vals = to_pred(view.value(M, replacement, np.random.default_rng(rng_seed)), pred_is_target)
    n = len(ks(F, step))
    out = {}
    for j, name in enumerate(names):
        dc, ic = vals[(2 * j) * n:(2 * j + 1) * n], vals[(2 * j + 1) * n:(2 * j + 2) * n]
        out[name] = {"deletion": dc, "insertion": ic, "auc_del": auc(dc), "auc_ins": auc(ic)}
    return out


def random_orders(F: int, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng([seed, 41])
    return [rng.permutation(F) for _ in range(n)]


def fidelity_for_instance(view, attributions: dict[str, np.ndarray], pred_is_target: bool, step: int,
                          n_random: int, modes: list[str], eval_seed: int = 0) -> tuple[list[dict], dict]:
    """Fidelity rows for every attribution of one instance (keys are e.g. 'kernel_shap|s0').

    Returns (rows, curves) — rows: one per (attribution key, mode); curves for the dashboard."""
    F = view.n_features
    rand = random_orders(F, n_random, eval_seed)
    repl = {"off_manifold": "zero", "on_manifold": "donor"}
    rows, curves = [], {}
    for mode in modes:
        orders = {f"__random{i}": o for i, o in enumerate(rand)}
        for key, a in attributions.items():
            s = a if pred_is_target else -a
            morf = np.argsort(-s, kind="stable")
            orders[key] = morf
            orders[f"{key}__lerf"] = morf[::-1]
        res = evaluate_orders(view, orders, step, repl[mode], pred_is_target, eval_seed)
        rd = np.mean([res[f"__random{i}"]["auc_del"] for i in range(n_random)])
        ri = np.mean([res[f"__random{i}"]["auc_ins"] for i in range(n_random)])
        curves[mode] = {"random_deletion": np.mean([res[f"__random{i}"]["deletion"] for i in range(n_random)], axis=0).tolist(),
                        "random_insertion": np.mean([res[f"__random{i}"]["insertion"] for i in range(n_random)], axis=0).tolist()}
        for key in attributions:
            r, lr = res[key], res[f"{key}__lerf"]
            rows.append({"key": key, "mode": mode, "auc_del": r["auc_del"], "auc_ins": r["auc_ins"],
                         "auc_del_random": rd, "auc_ins_random": ri, "auc_del_lerf": lr["auc_del"],
                         "auc_ins_lerf": lr["auc_ins"],
                         "fidelity": 0.5 * ((rd - r["auc_del"]) + (r["auc_ins"] - ri))})
            curves[mode][key] = {"deletion": r["deletion"].tolist(), "insertion": r["insertion"].tolist()}
    return rows, curves
