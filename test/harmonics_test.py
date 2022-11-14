import lal
import numpy as np
import pytest

from gwmemory import harmonics


@pytest.mark.parametrize("lm", harmonics.lmax_modes(4))
def test_spherical_harmonics_match_lal(lm):
    print(lm)
    for theta, phi in np.random.uniform(0, 2 * np.pi, (1000, 2)):
        theta /= 2
        diff = (
            harmonics.sYlm(ss=-2, ll=lm[0], mm=lm[1], theta=theta, phi=phi)
            - lal.SphHarm(lm[0], lm[1], theta, phi)[1]
        )
        assert abs(np.real(diff)) < 1e-6
        assert abs(np.imag(diff)) < 1e-6
