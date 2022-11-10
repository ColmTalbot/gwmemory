import pytest

from gwmemory import time_domain_memory

TEST_MODELS = [
    "IMRPhenomD",
    "SEOBNRv4",
    "NRSur7dq4",
    "NRHybSur3dq8",
    "NRSur7dq2",
    "MWM",
]


@pytest.mark.parametrize("model", TEST_MODELS)
def test_waveform_model_runs(model):
    q = 1
    total_mass = 60
    distance = 400
    spin_1 = [0, 0, 0]
    spin_2 = [0, 0, 0]
    inc = 1
    phase = 1
    time_domain_memory(
        model=model,
        q=q,
        total_mass=total_mass,
        distance=distance,
        spin_1=spin_1,
        spin_2=spin_2,
        inc=inc,
        phase=phase,
    )
