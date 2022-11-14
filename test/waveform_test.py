import os

import gwsurrogate as gws
import h5py
import numpy as np
import pytest
from gwtools import sxs_memory

import gwmemory
from gwmemory import frequency_domain_memory, time_domain_memory
from gwmemory.waveforms import Approximant, Surrogate, SXSNumericalRelativity

TEST_MODELS = [
    "IMRPhenomD",
    "IMRPhenomT",
    "IMRPhenomTHM",
    "IMRPhenomTPHM",
    "IMRPhenomXHM",
    "IMRPhenomXPHM",
    "MWM",
    "NRSur7dq2",
    "NRSur7dq4",
    "NRHybSur3dq8",
    "SEOBNRv4",
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


@pytest.mark.parametrize(
    ("model", "name"), ([Surrogate, "NRSur7dq4"], [Approximant, "IMRPhenomT"])
)
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
                data=np.vstack([times, h_osc[mode].real, h_osc[mode].imag]).T,
            )
    loaded = SXSNumericalRelativity("test_waveform.h5")
    mem1, times_1 = test.time_domain_memory()
    mem2, times_2 = loaded.time_domain_memory()
    assert np.allclose(mem1[(2, 0)], mem2[(2, 0)])
    assert np.allclose(times_1, times_2)
    os.remove("test_waveform.h5")


def test_memory_matches_sxs():
    model = gws.LoadSurrogate("NRHybSur3dq8")
    chi0 = [0, 0, 0.8]
    t = np.arange(-1000, 100, 0.01)
    t, h, dyn = model(8, chi0, chi0, times=t, f_low=0)
    h_mem, times = gwmemory.time_domain_memory(h_lm=h, times=t)
    h_mem_sxs, times_sxs = sxs_memory(h, t)
    modes = set(h_mem.keys()).intersection(h_mem_sxs.keys())

    for ii, mode in enumerate(modes):
        gwmem = h_mem[mode]
        sxsmem = h_mem_sxs[mode]
        overlap = (
            np.vdot(gwmem, sxsmem)
            / np.vdot(gwmem, gwmem) ** 0.5
            / np.vdot(sxsmem, sxsmem) ** 0.5
        )
        assert overlap.real > 1 - 1e-8
        assert abs(overlap.imag) < 1e-5
