# Paul Lasky
# Modified Colm Talbot
# This code package includes code for calculating the properties of quasinormal
# black hole modes

import os
import pkg_resources
import numpy as np

from .harmonics import sYlm
from . import utils


def freq_damping_obs(mass, spin, ell=2, mm=2, nn=0):
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
    f_lmn: float
        Frequency of mode in Hz
    tau_lmn: float
        Dampling time of mode in seconds
    """
    mass = utils.m_sol_to_geo(mass)
    omega, tau = freq_damping(mass, spin, ell, mm, nn)
    f_lmn = omega / (2 * np.pi)
    f_lmn = utils.freq_geo_to_Hz(f_lmn)
    tau = utils.time_geo_to_s(tau)
    return f_lmn, tau


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
    data_file = os.path.join(
        pkg_resources.resource_filename(__name__, "data"), "fitcoeffsWEB.dat"
    )
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


def amplitude(mass_1, mass_2, ell=2, mm=2):
    """
    Calculate the amplitude of the models relative to the 22 mode.
    see equations (5) -- (8) from Gossan et al. (2012) (note there is a
    difference between the arXiv paper and the real paper)

    Parameters
    ----------
    mass_1: float
        Mass of more massive BH
    mass_2: float
        Mass of less massive BH
    ell: int
        Spherical harmonic l
    mm: int
        Spherical harmonic m

    Returns
    -------
    float: Amplitude of the lm harmonic relative to the 22 mode.
    """

    # symmetric mass ratio
    nu = utils.m12_to_symratio(mass_1, mass_2)

    # all modes are normalised to the A22 mode
    amplitude_22 = 0.864 * nu

    if mm == 1:
        if ell == 2:
            return 0.52 * (1 - 4 * nu) ** (0.71) * amplitude_22
    elif ell == 2 & mm == 2:
        return amplitude_22
    elif ell == 3 & mm == 3:
        return 0.44 * (1 - 4 * nu) ** (0.45) * amplitude_22
    elif ell == 4 & mm == 4:
        return (5.4 * (nu - 0.22) ** 2 + 0.04) * amplitude_22
    else:
        print(f"Unknown mode ({ell}, {mm}) specified")


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


def hp_hx(time, m1, m2, ell, mm, iota, phi, phase):
    """
    Output hplus and hcross template:
    both calculated using eqns. (1) and (2) from Gossan et al.
    but with the normalisations ignored -- i.e., A_{ell m}=1, M_f = 1
    This is in geometric units.

    Parameters
    ----------
    time: array-like
        Array of times to evaluate template on, in geometric units.
    omega: float
        Angular frequency
    tau: float
        Damping time
    ell: int
        Spherical harmonic l
    mm: int
        Spherical harmonic m
    iota: float
        Angle between spin axis and line-of-sight to observer
    phi: float
        Azimuth angle of BH with respect to observer
    phase: float
        The third Euler angle

    Returns
    -------
    h_plus: array-like
        Plus polarisation
    h_cross: array-like
        Cross polarisation
    """
    ylm_plus = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) + (-1.0) ** ell * sYlm(
        -2, ll=ell, mm=-mm, theta=iota, phi=0
    )
    ylm_cross = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) - (-1.0) ** ell * sYlm(
        -2, ll=ell, mm=-mm, theta=iota, phi=0
    )

    amplitude_lm = amplitude(m1, m2, ell=ell, mm=mm)

    final_mass, jj = final_mass_spin(m1, m2)

    omega_lm, tau_lm = freq_damping(final_mass, jj, ell=ell, mm=mm, nn=0)

    h_plus = (
        final_mass
        * amplitude_lm
        * np.exp(-time / tau_lm)
        * ylm_plus
        * np.cos(omega_lm * time - mm * phi + phase)
    )

    h_cross = (
        final_mass
        * amplitude_lm
        * np.exp(-time / tau_lm)
        * ylm_cross
        * np.sin(omega_lm * time - mm * phi + phase)
    )

    return np.real(h_plus), np.real(h_cross)


def hp_hx_template(time, omega, tau, ell, mm, iota, phi, phase):
    """
    Output hplus and hcross template:
    both calculated using eqns. (1) and (2) from Gossan et al.
    but with the normalisations ignored -- i.e., A_{ell m}=1, M_f = 1
    This is in geometric units.

    Parameters
    ----------
    time: array-like
        Array of times to evaluate template on, in geometric units.
    omega: float
        Angular frequency
    tau: float
        Damping time
    ell: int
        Spherical harmonic l
    mm: int
        Spherical harmonic m
    iota: float
        Angle between spin axis and line-of-sight to observer
    phi: float
        Azimuth angle of BH with respect to observer
    phase: float
        The third Euler angle

    Returns
    -------
    h_plus: array-like
        Plus polarisation
    h_cross: array-like
        Cross polarisation
    """
    ylm_plus = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) + -(1 ** ell) * sYlm(
        -2, ll=ell, mm=-mm, theta=iota, phi=0
    )
    ylm_cross = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) - -(1 ** ell) * sYlm(
        -2, ll=ell, mm=-mm, theta=iota, phi=0
    )

    h_plus = np.exp(-time / tau) * ylm_plus * np.cos(omega * time - mm * phi + phase)

    h_cross = np.exp(-time / tau) * ylm_cross * np.sin(omega * time - mm * phi + phase)

    return np.real(h_plus), np.real(h_cross)
