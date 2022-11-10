import pytest

import numpy as np

from gwmemory import angles


@pytest.mark.parametrize("lm", [(2, 2), (2, -2), (3, 3), (3, 1)])
def test_numeric_gamma_agrees_with_analytic(lm):
    numeric = np.real(angles.gamma("22", f"{lm[0]}{lm[1]}"))
    analytic = np.array([angles.analytic_gamma((2, 2), lm, ell) for ell in range(2, len(numeric) + 2)])
    assert max(abs(numeric - analytic)) < 1e-4
    numeric = np.real(angles.gamma(f"{lm[0]}{lm[1]}", "22"))
    analytic = np.array([angles.analytic_gamma(lm, (2, 2), ell) for ell in range(2, len(numeric) + 2)])
    assert max(abs(numeric - analytic)) < 1e-4

