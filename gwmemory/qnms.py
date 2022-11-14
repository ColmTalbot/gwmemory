# Paul Lasky
# Modified Colm Talbot
# This code package includes code for calculating the properties of quasinormal
# black hole modes
from typing import Tuple

import numpy as np


def freq_damping(mass: float, spin: float, nn: int = 0) -> Tuple[float, float]:
    """
    Calculate the quasinormal mode freq and damping time for a black hole.
    This version uses OBSERVER'S UNITS.

    The magic numbers come from
    https://pages.jh.edu/eberti2/ringdown/fitcoeffsWEB.dat

    Parameters
    ----------
    mass: float
        BH mass in solar masses
    spin: float
        BH dimensionless spin
    nn: int, optional
        QNM harmonic, default=0 is the fundamental

    Returns
    -------
    omega_lmn: float
        Angular frequency of mode in geometric units
    tau_lmn: float
        Damping time of mode in geometric units
    """

    f1 = [1.5251, 1.3673, 1.3223][nn]
    f2 = [-1.1568, -1.0260, -1.0257][nn]
    f3 = [0.1292, 0.1628, 0.1860][nn]

    q1 = [0.7, 0.1, -0.1][nn]
    q2 = [1.4187, 0.5436, 0.4206][nn]
    q3 = [-0.4990, 0.4731, -0.4256][nn]

    # dimensionaless frequency
    f_lmn = f1 + f2 * (1.0 - spin) ** f3

    # angular frequency
    omega_lmn = f_lmn / mass

    # quality factor
    Q_lmn = q1 + q2 * (1.0 - spin) ** q3

    # damping time
    tau_lmn = 2.0 * Q_lmn / omega_lmn

    return omega_lmn, tau_lmn


def final_mass_spin(mass_1: float, mass_2: float) -> Tuple[float, float]:
    """
    Given initial total mass and mass ratio, calculate the final mass, M_f
    and dimensionless spin parameter, jj,
    using fits in Buonanno et al. (2007) -- see the caption of their Table I.

    Parameters
    ----------
    mass_1: float
        Mass of more massive black hole.
    mass_2: float
        Mass of less massive black hole.

    Returns
    -------
    final_mass: float
        Remnant mass
    jj: float
        Remnant dimensionless spin
    """

    mass_1 = float(mass_1)
    mass_2 = float(mass_2)

    eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2
    total_mass = mass_1 + mass_2

    # expansion coefficients for the mass
    m_f1 = 1.0
    m_f2 = (np.sqrt(8.0 / 9.0) - 1.0) * eta
    m_f3 = -0.498 * eta**2

    # final mass
    final_mass = total_mass * (m_f1 + m_f2 + m_f3)

    # expansion coefficients for spin parameter
    a_f1 = np.sqrt(12.0) * eta
    a_f2 = -2.9 * eta**2

    # final spin parameter -- dimensions of mass
    a_f = final_mass * (a_f1 + a_f2)

    # dimensionless spin
    jj = a_f / final_mass

    return final_mass, jj
