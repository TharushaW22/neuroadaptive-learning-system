"""Step 4: instance selection and applicability screening."""

import numpy as np
import pandas as pd

from xaicomp.adapters.base import Capabilities
from xaicomp.data.instances import select_control_instances, select_instances
from xaicomp.screening.applicability import CHECK_COLUMNS, screen


def _toy_manifest():
    rows = []
    for s in (1, 2, 3):
        for c in ("lo", "hi"):
            for e in range(0, 20, 2):
                rows.append({"instance_id": f"s{s:02d}_{c}_e{e:02d}", "subject_id": s, "condition": c, "fold": s % 2,
                             "y": int(c == "hi")})
    return pd.DataFrame(rows)


def test_selection_is_stratified_and_deterministic():
    m = _toy_manifest()
    a = select_instances(m, "all", 4, 7)
    b = select_instances(m, "all", 4, 7)
    assert a["instance_id"].tolist() == b["instance_id"].tolist()
    assert (a.groupby(["subject_id", "condition"]).size() == 2).all()
    odd = select_instances(m, [1], 3, 7)
    assert len(odd) == 3 and set(odd["condition"]) == {"lo", "hi"}


def test_control_is_subset_of_primary():
    m = _toy_manifest()
    sel = select_instances(m, "all", 2, 1)
    ids = select_control_instances(sel, 1, 1)
    assert set(ids) <= set(sel["instance_id"]) and len(ids) == 2           # one per fold


def test_screening_excludes_ig_for_trees_and_dice_anchors_everywhere():
    budgets = {"kernel_shap": {"nsamples": 100, "k_donors": 2}, "conditional_shap": {"nsamples": 100, "k_donors": 2},
               "lime": {"num_samples": 100, "k_donors": 2}, "ig": {"n_steps": 8, "k_donors": 2}, "ig_zero": {"n_steps": 8}}
    deep = screen("deep", Capabilities(True, 84, "deep"), budgets, 10.0, 600)
    tree = screen("tree", Capabilities(False, 6, "tree ensemble"), budgets, 10.0, 600)
    v = deep.set_index("method")["verdict"]
    assert v["ig"] == "applicable" and v["dice"] == "excluded" and v["anchors"] == "excluded"
    t = tree.set_index("method")
    assert t.loc["ig", "verdict"] == "excluded" and t.loc["ig", "differentiable"] == "fail"
    assert t.loc["kernel_shap", "verdict"] == "applicable"
    assert deep.set_index("method").loc["ig_zero", "repeatable"] == "caution"
    assert all(c in deep for c in CHECK_COLUMNS) and len(CHECK_COLUMNS) == 8
    slow = screen("deep", Capabilities(True, 84, "deep"), budgets, 10_000.0, 1)
    assert slow.set_index("method").loc["kernel_shap", "compute_budget"] == "fail"


def test_manifest_on_real_data(adapter):
    from xaicomp.data.instances import build_manifest
    m = build_manifest(adapter)
    assert len(m) == 3264
    assert m["p"].between(0, 1).all() and m["confidence"].between(0.5, 1).all()
    assert m["in_sample"].all() == adapter.in_sample
    assert set(["subject_id", "fold", "y", "frac_bad", "model_source"]) <= set(m.columns)
