# Paul Lasky
# Modified Colm Talbot
# This code package includes code for calculating the properties of quasinormal
# black hole modes

from pathlib import Path
import numpy as np


def freq_damping(mass, spin, ell=2, mm=2, nn=0):
    """
    Calculate the quasinormal mode freq and damping time for a black hole.
    This version uses OBSERVER'S UNITS.

    Parameters
    ----------
    mass: float
        BH mass in solar masses
    spin: float
        BH dimensionless spin
    ell: int, optional
        Spherical harmonic l, default=2
    mm: int, optional
        Spherical harmonic m, default=2
    nn: int, optional
        QNM harmonic, default=0 is the fundamental

    Returns
    -------
    omega_lmn: float
        Angular frequency of mode in geometric units
    tau_lmn: float
        Dampling time of mode in geometric units
    """
    data_file = str(Path(__file__).parent / "data" / "fitcoeffsWEB.dat")
    data = np.loadtxt(data_file)

    ell_data = data[:, 0].astype(int)
    mm_data = data[:, 1].astype(int)
    nn_data = data[:, 2].astype(int)

    cond = (ell_data == ell) & (mm_data == mm) & (nn_data == nn)

    f1 = data[cond, 3][0]
    f2 = data[cond, 4][0]
    f3 = data[cond, 5][0]

    q1 = data[cond, 6][0]
    q2 = data[cond, 7][0]
    q3 = data[cond, 8][0]

    # dimensionaless frequency
    f_lmn = f1 + f2 * (1.0 - spin) ** f3

    # angular frequency
    omega_lmn = f_lmn / mass

    # quality factor
    Q_lmn = q1 + q2 * (1.0 - spin) ** q3

    # damping time
    tau_lmn = 2.0 * Q_lmn / omega_lmn

    return omega_lmn, tau_lmn


def final_mass_spin(mass_1, mass_2):
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
    m_f3 = -0.498 * eta ** 2

    # final mass
    final_mass = total_mass * (m_f1 + m_f2 + m_f3)

    # expansion coefficients for spin parameter
    a_f1 = np.sqrt(12.0) * eta
    a_f2 = -2.9 * eta ** 2

    # final spin parameter -- dimensions of mass
    a_f = final_mass * (a_f1 + a_f2)

    # dimensionless spin
    jj = a_f / final_mass

    return final_mass, jj
