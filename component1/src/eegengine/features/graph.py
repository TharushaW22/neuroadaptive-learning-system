"""Graph-theory features of the PLV brain network (family prefix `graph_`), using networkx.

Each epoch/band PLV matrix is a weighted graph: 14 electrodes = nodes, PLV = edge weight.
Thresholding strategies (configurable):
  proportional : keep the strongest `proportion` of edges (e.g. top 30 %), weights kept
  absolute     : keep edges with PLV >= `absolute`
  none         : fully connected weighted graph
Per node : strength (sum of weights), clustering coefficient (weighted, Onnela), betweenness
           centrality (distance = 1 / weight), eigenvector centrality (weighted).
Per graph: global efficiency (weighted: mean of 1 / shortest-path distance), density
           (fraction of possible edges present; for "none" the mean weight), and graph entropy
           (Shannon entropy of the normalised node-strength distribution).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

NODE_METRICS = ("strength", "clustering", "betweenness", "eigenvector")
GLOBAL_METRICS = ("efficiency", "density", "entropy")


def threshold_matrix(W: np.ndarray, method: str = "proportional", proportion: float = 0.3,
                     absolute: float = 0.5) -> np.ndarray:
    """Return a thresholded copy of a symmetric weight matrix (diagonal set to 0)."""
    A = W.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    if method == "none":
        return A
    if method == "absolute":
        A[A < absolute] = 0.0
        return A
    if method == "proportional":
        iu = np.triu_indices(A.shape[0], k=1)
        vals = A[iu]
        k = max(1, int(round(proportion * len(vals))))
        cut = np.sort(vals)[-k]
        A[A < cut] = 0.0
        return A
    raise ValueError(f"Unknown threshold method {method!r}")


def graph_metrics(A: np.ndarray, method: str) -> dict[str, np.ndarray | float]:
    """Metrics of one thresholded weighted adjacency matrix A (n x n)."""
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    strength = A.sum(axis=1)
    clustering = np.array([v for _, v in sorted(nx.clustering(G, weight="weight").items())])

    # distances = 1 / weight for path-based metrics (strong coupling = short path)
    for _, _, d in G.edges(data=True):
        d["distance"] = 1.0 / d["weight"]
    betweenness = np.array([v for _, v in sorted(nx.betweenness_centrality(G, weight="distance").items())])

    # eigenvector centrality = leading eigenvector of the symmetric weight matrix
    vals, vecs = np.linalg.eigh(A)
    ev = np.abs(vecs[:, -1])
    ev = ev / (np.linalg.norm(ev) + 1e-20)

    with np.errstate(divide="ignore"):
        D = np.where(A > 0, 1.0 / A, 0.0)
    sp = shortest_path(D, method="D", directed=False)          # inf if disconnected
    off = ~np.eye(n, dtype=bool)
    efficiency = float(np.mean(1.0 / sp[off]))                # 1/inf = 0 for unreachable pairs

    n_possible = n * (n - 1) / 2
    density = float(A[np.triu_indices(n, 1)].mean()) if method == "none" else float((A > 0).sum() / 2 / n_possible)
    p = strength / (strength.sum() + 1e-20)
    entropy = float(-(p * np.log(p + 1e-20)).sum() / np.log(n))
    return {"strength": strength, "clustering": clustering, "betweenness": betweenness,
            "eigenvector": ev, "efficiency": efficiency, "density": density, "entropy": entropy}


def graph_features(plv: np.ndarray, band_names: list[str], ch_names: list[str], method: str = "proportional",
                   proportion: float = 0.3, absolute: float = 0.5) -> dict[str, np.ndarray]:
    """plv: (N, bands, ch, ch) -> {column: (N,)} with graph_<band>_<metric>[_<ch>] names."""
    N, nb, nch, _ = plv.shape
    node = {m: np.zeros((N, nb, nch)) for m in NODE_METRICS}
    glob = {m: np.zeros((N, nb)) for m in GLOBAL_METRICS}
    for e in range(N):
        for b in range(nb):
            res = graph_metrics(threshold_matrix(plv[e, b], method, proportion, absolute), method)
            for m in NODE_METRICS:
                node[m][e, b] = res[m]
            for m in GLOBAL_METRICS:
                glob[m][e, b] = res[m]
    cols = {}
    for b, band in enumerate(band_names):
        for m in GLOBAL_METRICS:
            cols[f"graph_{band}_{m}"] = glob[m][:, b]
        for m in NODE_METRICS:
            for k, ch in enumerate(ch_names):
                cols[f"graph_{band}_{m}_{ch}"] = node[m][:, b, k]
    return cols
