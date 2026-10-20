"""Steps 12-13: TOPSIS, SMAA, criteria rules, C4 export schema."""

import json

import jsonschema
import numpy as np
import pandas as pd
import pytest

from xaicomp.export.schema import EXPORT_SCHEMA, validate
from xaicomp.ranking.criteria import build_values, decide, final_criteria, scale_values
from xaicomp.ranking.smaa import smaa, smaa_multi_seed
from xaicomp.ranking.topsis import topsis

RULES = {"min_icc_keep": 0.5, "full_weight_icc": 0.75, "require_monotonic": True, "monotonic_min_abs_rho": 0.8,
         "redundancy_rho": 0.9, "must_fail_control": ["fidelity_on"]}


def test_topsis_dominance():
    V = np.array([[0.9, 1.0], [0.5, 2.0], [0.4, 3.0]])           # benefit, cost
    cl = topsis(V, np.array([0.5, 0.5]), np.array([True, False]))
    assert cl[0] == pytest.approx(1.0) and cl[2] == pytest.approx(0.0) and cl[0] > cl[1] > cl[2]


def test_smaa_dominant_method_always_first_and_stable():
    rng = np.random.default_rng(0)
    Vb = np.stack([np.array([[0.9, 0.8], [0.5, 0.5], [0.4, 0.3]]) + rng.normal(0, 0.01, (3, 2)) for _ in range(100)])
    r = smaa(Vb, np.ones(2), np.array([True, True]), 2000, 1.0, 1)
    assert r["p_rank1"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(r["rank_acceptability"].sum(axis=0), 1.0)
    m = smaa_multi_seed(Vb, np.ones(2), np.array([True, True]), 2000, 1.0, [1, 2, 3])
    assert m["mc_max_abs_diff"] < 0.01


def test_smaa_tradeoff_splits_rank1():
    Vb = np.repeat(np.array([[[0.9, 0.1], [0.1, 0.9]]]), 10, axis=0)
    r = smaa(Vb, np.ones(2), np.array([True, True]), 20000, 1.0, 0)
    assert r["p_rank1"][0] == pytest.approx(0.5, abs=0.03)


def test_criteria_rules():
    icc = pd.DataFrame({"criterion": ["fidelity_on", "stability", "sparsity", "cost_seconds"], "icc": [0.9, 0.3, 0.6, np.nan]})
    deg = pd.DataFrame({"criterion": ["fidelity_on", "stability", "sparsity"], "rho_level_vs_value": [-1.0, -1.0, -0.2],
                        "monotonic_decrease": [True, True, False]})
    col = pd.DataFrame({"criterion": ["fidelity_on"], "collapses": [True]})
    d = decide(["fidelity_on", "stability", "sparsity", "cost_seconds"], icc, deg, col, [["fidelity_on"]], RULES).set_index("criterion")
    assert d.loc["fidelity_on", "decision"] == "keep"
    assert d.loc["stability", "decision"] == "drop"                   # ICC 0.3
    assert d.loc["sparsity", "decision"] == "drop"                    # not monotonic
    assert d.loc["cost_seconds", "decision"] == "keep" and "not assessable" in d.loc["cost_seconds", "reasons"]
    col2 = pd.DataFrame({"criterion": ["fidelity_on"], "collapses": [False]})
    assert decide(["fidelity_on"], icc, deg, col2, [], RULES).iloc[0]["decision"] == "drop"
    icc3 = icc.assign(icc=[0.6, 0.9, 0.9, np.nan])
    assert decide(["fidelity_on"], icc3, deg, col, [], RULES).iloc[0]["weight_factor"] == pytest.approx(0.6)


def test_merging_and_scaling():
    dec = pd.DataFrame({"criterion": ["fidelity_on", "fidelity_off", "cost_seconds"], "decision": ["keep"] * 3,
                        "weight_factor": [1.0, 0.6, 1.0]})
    fin = final_criteria(dec, [["fidelity_on", "fidelity_off"], ["cost_seconds"]])
    assert fin[0]["members"] == ["fidelity_on", "fidelity_off"] and fin[0]["factor"] == pytest.approx(0.8)
    assert fin[1]["benefit"] is False
    B = np.array([[[0.2, 0.0, 5.0]]])
    V = build_values(B, ["fidelity_on", "fidelity_off", "cost_seconds"], fin)
    assert V[0, 0, 0] == pytest.approx((0.6 + 0.5) / 2) and V[0, 0, 1] == 5.0
    assert scale_values(np.array([-2.0]), "stability")[0] == 0.0


def _doc():
    w = {"context_id": "c", "instance_id": "i", "subject_id": 1, "method": "lime",
         "prediction": {"target_class": "high", "p_target": 0.8, "predicted_class": "high", "confidence": 0.8},
         "top_k": {lvl: [{"name": "x", "attribution": 0.1, "direction": "towards_target"}] for lvl in ("region", "band", "channel_x_band")},
         "trust": {"level": "low", "score": 0.2, "components": {}}, "in_sample": True}
    c = {"context_id": "c", "run_type": "PILOT", "in_sample": True, "recommended_method": "lime", "p_rank1": {"lime": 0.6},
         "runner_up": "ig", "practically_equivalent_to_runner_up": False,
         "equivalence": {"p_recommended_better": 0.5, "p_rope": 0.3, "p_runner_up_better": 0.2, "rope": 0.05},
         "criteria_used": ["fidelity_on"]}
    return {"schema_version": "1.0", "generated": "now", "run": {"name": "pilot", "run_type": "PILOT"},
            "contexts": [c], "windows": [w]}


def test_export_schema_accepts_valid_and_rejects_invalid():
    validate(_doc())
    bad = _doc()
    bad["contexts"][0]["p_rank1"]["lime"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        validate(bad)
    bad2 = _doc()
    del bad2["windows"][0]["trust"]
    with pytest.raises(jsonschema.ValidationError):
        validate(bad2)


def test_real_export_validates_if_present():
    from xaicomp.paths import RESULTS_DIR
    found = list(RESULTS_DIR.glob("*/component4_export.json"))
    if not found:
        pytest.skip("no run exported yet")
    for p in found:
        doc = json.loads(p.read_text())
        jsonschema.validate(doc, EXPORT_SCHEMA)
        for c in doc["contexts"]:
            assert abs(sum(c["p_rank1"].values()) - 1) < 1e-6
