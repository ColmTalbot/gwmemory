import h5py
import os

import numpy as np
import pytest

from gwmemory import time_domain_memory, frequency_domain_memory
from gwmemory.waveforms import Approximant, Surrogate, SXSNumericalRelativity

TEST_MODELS = [
    "IMRPhenomD",
    "IMRPhenomT",
    "SEOBNRv4",
    "NRSur7dq4",
    "NRHybSur3dq8",
    "NRSur7dq2",
    "MWM",
]

PARAMS = dict(
    total_mass=60,
    q=1,
    distance=400,
    spin_1=[0, 0, 0],
    spin_2=[0, 0, 0],
    inc=1,
    phase=1,
    minimum_frequency=20,
    sampling_frequency=4096,
)


@pytest.mark.parametrize("model", TEST_MODELS)
def test_waveform_model_runs(model):
    mem, _ = time_domain_memory(model, **PARAMS)
    assert mem["plus"][-1] > 1e-22


@pytest.mark.parametrize("model", TEST_MODELS)
def test_waveform_multiple_runs(model):
    """
    Checks an issue reported in
    https://github.com/ColmTalbot/gwmemory/pull/6
    """
    times = np.linspace(-0.04, -0.03, 10001)
    mem_1, times_1 = time_domain_memory(model, **PARAMS, times=times)
    mem_2, times_2 = time_domain_memory(model, **PARAMS, times=times)
    assert np.allclose(times_1, times_2)
    for mode in mem_1:
        assert np.allclose(mem_1[mode], mem_2[mode])


@pytest.mark.parametrize(("model", "name"), ([Surrogate, "NRSur7dq4"], [Approximant, "IMRPhenomT"]))
def test_minimal_arguments(model, name):
    """
    Test calculating memory with default arguments.

    Because no scale is provided, we manually translate to geometric units as
    some models don't do that transformation internally.
    """
    test = model(q=1, name=name)
    mem, _ = test.time_domain_memory()
    assert mem[(2, 0)][-1] * test.h_to_geo > 1e-2


def test_fd_memory_runs():
    """
    This function doesn't actually do much, so the test is a little feeble.
    """
    mem, _ = frequency_domain_memory("IMRPhenomT", **PARAMS)
    assert [key in mem for key in ["plus", "cross"]]


def test_nr_waveform():
    """
    Verify that writing/reading an existing waveform to the NR format gives
    the same memory.
    """
    test = Surrogate(q=1)
    h_osc, times = test.time_domain_oscillatory()
    with h5py.File("test_waveform.h5", "w") as ff:
        grp = ff.create_group("OutermostExtraction.dir")
        for mode in h_osc:
            grp.create_dataset(
                f"Y_l{mode[0]}_m{mode[1]}.dat",
                data=np.vstack([times, h_osc[mode].real, h_osc[mode].imag]).T
            )
    loaded = SXSNumericalRelativity("test_waveform.h5")
    mem1, times_1 = test.time_domain_memory()
    mem2, times_2 = loaded.time_domain_memory()
    assert np.allclose(mem1[(2, 0)], mem2[(2, 0)])
    assert np.allclose(times_1, times_2)
    os.remove("test_waveform.h5")
