import numpy as np
from algorithms.linucb import LinUCB
from algorithms.qlearning import QLearning
from algorithms.bkt import BKT
from algorithms.topsis import TOPSIS


def test_shapes_and_update():
    for Algo in (LinUCB, QLearning, BKT, TOPSIS):
        a = Algo(dim=23) if Algo is LinUCB else Algo()
        ctx = np.random.default_rng(0).random(23).astype("float32")
        s, p = a.select(ctx)
        assert isinstance(s, str) and isinstance(p, str)
        a.update(ctx, s, 0.5)


def test_fresh_bandit_not_first_arm():
    outcomes = set()
    for seed in range(20):
        a = LinUCB(dim=5, seed=seed)
        s, _ = a.select(np.zeros(5))
        outcomes.add(s)
    assert len(outcomes) > 1