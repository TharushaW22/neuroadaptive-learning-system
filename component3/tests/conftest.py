import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def _data_ready() -> bool:
    from xaicomp.data.eeg_data import data_available
    return data_available()


@pytest.fixture(scope="session")
def data_ready():
    if not _data_ready():
        pytest.skip("blocked: STEW / processed data missing (run scripts/build_data.py)")
    return True


@pytest.fixture(scope="session")
def adapter(data_ready):
    import torch
    from xaicomp.adapters.eeg_component1 import EEGComponent1Adapter
    from xaicomp.config import load_context
    torch.set_num_threads(2)
    return EEGComponent1Adapter(load_context("eeg_T1_A8"), threads=2)
