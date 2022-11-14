import numpy as np
import pytest
from sxs.waveforms import WaveformModes
from sxs.waveforms.memory import Dinverse

from gwmemory import angles


@pytest.mark.parametrize("lm", [(2, 2), (2, -2), (3, 3), (3, 1)])
def test_numeric_gamma_agrees_with_analytic(lm):
    numeric = np.real(angles.gamma("22", f"{lm[0]}{lm[1]}"))
    analytic = np.array(
        [angles.analytic_gamma((2, 2), lm, ell) for ell in range(2, len(numeric) + 2)]
    )
    assert max(abs(numeric - analytic)) < 1e-4
    numeric = np.real(angles.gamma(f"{lm[0]}{lm[1]}", "22"))
    analytic = np.array(
        [angles.analytic_gamma(lm, (2, 2), ell) for ell in range(2, len(numeric) + 2)]
    )
    assert max(abs(numeric - analytic)) < 1e-4


def test_memory_factor_matches_sxs():
    """
    The analytic gamma calculation should agree with the combination of two
    contributions. This one is from Eq. (12) of arXiv:2011.01309
    """
    h_test = WaveformModes(
        np.ones((2, 21)),
        ell_min=2,
        ell_max=4,
        modes_axis=1,
        time_axis=0,
        time=np.array([0, 1]),
    )
    h_test._metadata["spin_weight"] = 0
    corrected = Dinverse(h_test).ethbar.ethbar.ndarray[0, 3:] / 2
    for ell in [2, 3, 4]:
        assert corrected[h_test.index(ell, 0)] == angles.memory_correction(ell)
