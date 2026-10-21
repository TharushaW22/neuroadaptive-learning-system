"""Step 1: STEW -> processed data with C1's library; counts must match C1's docs/labels.md."""

import re

import numpy as np
import pytest

from xaicomp._bootstrap import C1_ROOT


@pytest.fixture(scope="module")
def data(data_ready):
    from xaicomp.data.eeg_data import load_data
    return load_data()


def test_epoch_count_and_shape(data):
    assert data.X.shape == (7104, 14, 512)
    assert data.continuous.shape == (96, 14, 19200)
    assert (data.meta.groupby(["subject_id", "condition"]).size() == 74).all()


def test_label_counts_match_c1_labels_doc(data):
    doc = (C1_ROOT / "docs" / "labels.md").read_text(encoding="utf-8")
    t1 = re.search(r"T1 workload_binary \| epochs \| low (\d+), high (\d+)", doc)
    t3 = re.search(r"T3 workload_within_task \| epochs \| moderate (\d+), high (\d+)", doc)
    m = data.meta
    assert (m["y_binary"] == 0).sum() == int(t1.group(1)) and (m["y_binary"] == 1).sum() == int(t1.group(2))
    assert (m["y_within"] == 0).sum() == int(t3.group(1)) and (m["y_within"] == 1).sum() == int(t3.group(2))


def test_epochs_are_slices_of_continuous_signal(data):
    for row in [0, 1000, 5555, 7103]:
        r = data.meta.iloc[row]
        s = int(r["epoch_index"]) * 256
        np.testing.assert_array_equal(data.X[row], data.continuous[int(r["rec_index"]), :, s:s + 512])


def test_rec_index_matches_keys(data):
    for _, r in data.meta.drop_duplicates("rec_index").iterrows():
        assert data.rec_keys[int(r["rec_index"])] == f"sub{int(r['subject_id']):02d}_{r['condition']}"


def test_config_hash_is_c1s():
    from eegengine.preprocess.pipeline import config_hash
    from xaicomp.config import load_c1_config
    assert config_hash(load_c1_config("preprocess.yaml")) == "6d7dd612"
