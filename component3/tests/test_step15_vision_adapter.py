"""Step 15 (vision): interpretable space, replacement strategies, adapter and the four explainers on
shape-level toy fixtures (random numbers + a random GRU; tests only, nothing written to results/)."""

import numpy as np
import pytest
import torch

from fixtures_vision import GROUPS, T, toy_adapter, toy_ctx
from xaicomp.models.vision_c2 import C2_FEATURE_COLUMNS
from xaicomp.replacement.vision_strategies import compose_frames, gaussian_suitability, segment_means
from xaicomp.representation.vision_space import VisionSpace, segment_names, space_from_info


@pytest.fixture(scope="module")
def ad():
    torch.set_num_threads(1)
    return toy_adapter()


# ------------------------------------------------------------------ space
def test_space_layout_and_groups():
    sp = VisionSpace(C2_FEATURE_COLUMNS, segment_names(4, 20, 5), GROUPS)
    assert sp.n_features == 60 and sp.segments == ["0-1 s", "1-2 s", "2-3 s", "3-4 s"]
    assert sp.feature_names[0] == "ear_left|0-1 s" and sp.index("gaze_v", "2-3 s") == 6 * 4 + 2
    # every feature in exactly one group; every level partitions the 60 features
    assert sorted(f for fs in GROUPS.values() for f in fs) == sorted(C2_FEATURE_COLUMNS)
    for lvl in ("feature", "segment", "group", "group_x_segment", "feature_x_segment"):
        idx = sorted(i for g in sp.groups(lvl).values() for i in g)
        assert idx == list(range(60)), lvl
    v = np.arange(60.0)
    agg, names = sp.aggregate(v, "group")
    assert agg.sum() == v.sum() and names == list(GROUPS)
    assert space_from_info(sp.describe()).feature_names == sp.feature_names


def test_space_rejects_bad_groups():
    with pytest.raises(ValueError):
        VisionSpace(C2_FEATURE_COLUMNS, ["a"], {"x": ["not_a_feature"]})
    with pytest.raises(ValueError):
        VisionSpace(C2_FEATURE_COLUMNS, ["a"], {"x": ["gaze_h"], "y": ["gaze_h"]})


# ------------------------------------------------------------------ replacement
def test_compose_frames_zero_and_donor():
    rng = np.random.default_rng(0)
    x, d = rng.normal(size=(15, 4, 5)).astype(np.float32), rng.normal(size=(15, 4, 5)).astype(np.float32)
    m = np.ones((1, 60), bool)
    m[0, 5] = False                                     # feature 1, segment 1
    z = compose_frames(x, m, None)[0]
    assert np.all(z[1, 1] == 0) and np.allclose(z[0], x[0])          # zero = training mean
    dn = compose_frames(x, m, d)[0]
    assert np.allclose(dn[1, 1], d[1, 1]) and np.allclose(dn[2], x[2])
    sh = np.zeros((1, 15, 4), np.float32)
    sh[0, 1, 1] = 2.0
    assert np.allclose(compose_frames(x, m, d, sh)[0][1, 1], d[1, 1] + 2.0)


def test_gaussian_suitability_decides_from_data():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(600, 6))
    Zg = A @ rng.normal(size=(6, 6))                     # correlated Gaussian -> calibrated
    good = gaussian_suitability(Zg, groups=np.repeat(np.arange(60), 10))
    assert good["mode"] == "gaussian", good
    Zd = (rng.random((600, 6)) < 0.1).astype(float)      # binary -> discrete -> empirical
    bad = gaussian_suitability(Zd, groups=np.repeat(np.arange(60), 10))
    assert bad["mode"] == "empirical" and bad["discrete_share"] == 1.0


# ------------------------------------------------------------------ adapter
def test_instances_are_test_split_with_int_subjects(ad):
    inst = ad.instances()
    assert set(inst["split"]) == {"test"} and inst["subject_id"].dtype.kind == "i"
    assert set(inst["condition"]) <= {"low", "high"} and set(inst["fold"]) == {0}
    assert ad.in_sample is False                          # toy split is subject-disjoint
    assert inst["frac_bad"].max() == pytest.approx(0.25)  # the padded clip


def test_overlap_makes_everything_in_sample():
    a = toy_adapter(overlap=True)
    assert a.in_sample is True and "IN-SAMPLE" in a.run_info_extra()["label"]


def test_value_function_consistency(ad):
    iid = ad.instances()["instance_id"].iloc[1]
    v = ad.view(iid, 2, seed=0)
    full = np.ones((1, 60), bool)
    p_inst = ad.predict_instances([iid])[0, ad.target_idx]
    assert v.fx() == pytest.approx(p_inst, abs=1e-6)
    for repl in ("zero", "donor", "conditional", "empirical"):
        assert v.value(full, repl)[0] == pytest.approx(v.fx(), abs=1e-6), repl
    # donors come from training subjects only
    assert set(ad.data.windows.splits[v.donor_pos]) == {"train"}


def test_conditional_targets_are_hit_exactly(ad):
    iid = ad.instances()["instance_id"].iloc[2]
    v = ad.view(iid, 3, seed=1)
    m = np.random.default_rng(0).random((4, 60)) < 0.5
    from xaicomp.replacement.vision_strategies import conditional_shifts
    shifts, targets = conditional_shifts(m, v.z_x, v.donor_z, ad.gaussian(), np.random.default_rng(1))
    w = compose_frames(v.x, m[:1], v.donors[0], shifts[:1, 0].reshape(1, 15, 4))
    got = segment_means(w)[0]
    A = ~m[0]
    np.testing.assert_allclose(got[A], targets[0, 0, A], atol=1e-5)
    np.testing.assert_allclose(got[~A], v.z_x[~A], atol=1e-6)


def test_noise_view_keeps_binary_feature_and_padding(ad):
    pad_iid = ad.instances().sort_values("frac_bad")["instance_id"].iloc[-1]
    v0, vn = ad.view(pad_iid, 0, 0), ad.view(pad_iid, 0, 0, noise_seed=0, noise_scale=0.1)
    d = vn.x - v0.x
    assert np.all(d[C2_FEATURE_COLUMNS.index("blink_event")] == 0)
    assert np.all(d[:, -1, :] == 0)                       # last segment is padding in the toy clip
    assert np.abs(d).max() > 0


def test_local_feature_correlation_in_range(ad):
    r = ad.local_feature_correlation(ad.instances()["instance_id"].iloc[0])
    assert 0 <= r <= 1


def test_randomised_control_changes_predictions():
    a = toy_adapter()
    b = toy_adapter(ctx={"weights": "randomised", "randomisation_seed": 99})
    ids = a.instances()["instance_id"].tolist()[:5]
    assert not np.allclose(a.predict_instances(ids), b.predict_instances(ids))


def test_segments_must_divide_window():
    from xaicomp.adapters.vision_component2 import VisionComponent2Adapter
    from fixtures_vision import toy_data, toy_model
    with pytest.raises(ValueError):
        VisionComponent2Adapter(toy_ctx(representation={"segments": 3}), data=toy_data(), model=toy_model(), threads=1)


# ------------------------------------------------------------------ explainers on the vision adapter
@pytest.mark.parametrize("method,budget", [("kernel_shap", {"nsamples": 200, "k_donors": 2}),
                                           ("conditional_shap", {"nsamples": 200, "k_donors": 2}),
                                           ("lime", {"num_samples": 200, "k_donors": 2}),
                                           ("ig", {"n_steps": 16, "k_donors": 2}),
                                           ("ig_zero", {"n_steps": 16})])
def test_explainers_run_and_check(ad, method, budget):
    from xaicomp.config import load_c3_config
    from xaicomp.explainers.registry import NEEDS_DONORS, explain

    cfg = load_c3_config("explainers.yaml")["methods"][method]
    iid = ad.instances()["instance_id"].iloc[3]
    v = ad.view(iid, budget.get("k_donors", 0) if method in NEEDS_DONORS else 0, seed=0)
    e = explain(method, v, 0, budget, cfg, ad.target_class)
    assert e.values.shape == (60,) and np.all(np.isfinite(e.values))
    if method in ("kernel_shap", "conditional_shap"):
        assert e.extras["efficiency_error"] < 1e-6
    if method.startswith("ig"):
        assert e.extras["completeness_error_rel"] < 0.05 or e.extras["completeness_error"] < 1e-3
    if method == "conditional_shap":
        assert e.extras["replacement"] in ("conditional", "empirical")


def test_fidelity_runs_on_vision_view(ad):
    from xaicomp.metrics.fidelity import fidelity_for_instance
    iid = ad.instances()["instance_id"].iloc[4]
    v = ad.view(iid, 2, 999)
    a = np.random.default_rng(0).normal(size=60)
    rows, curves = fidelity_for_instance(v, {"x|s0": a}, v.fx() >= 0.5, 5, 3, ["off_manifold", "on_manifold"])
    assert len(rows) == 2 and all(np.isfinite(r["fidelity"]) for r in rows)


def test_window_length_constant():
    assert T == 20
