# Paul Lasky
# This code package includes code for calculating the properties of quasinormal
# black hole modes

import numpy as np
from .harmonics import sYlm
from . import utils
import os
import pkg_resources


def freq_damping_obs(MM, jj, ell=2, mm=2, nn=0):
    """
    Calculate the quasinormal mode freq and damping time for a black hole.
    This version uses OBSERVER'S UNITS.
    MM = mass  (in msun)
    jj = dimensionless angular momentum parameter
    ell,mm = mode ... (2,2) is the default
    nn = tone ... 0 is the default
    returns:  f_lmn, tau_lmn
    f_lmn is observed frequency in Hz
    tau_lmn is the daming time in seconds
    """
    MM = utils.m_sol_to_geo(MM)
    omega, tau = freq_damping(MM, jj, ell, mm, nn)
    f = omega / (2*np.pi)
    f = utils.freq_geo_to_Hz(f)
    tau = utils.time_geo_to_s(tau)
    return f, tau


def freq_damping(MM, jj, ell=2, mm=2, nn=0):
    """
    Calculate the quasinormal mode freq and damping time for a black hole.
    MM = mass  (in geometric units)
    jj = dimensionless angular momentum parameter
    ell,mm = mode ... (2,2) is the default
    nn = tone ... 0 is the default
    returns:  omega_lmn, tau_lmn
    omega_lmn is ANGULAR frequency in GEOMETRIC units
    tau_lmn is the daming time in geometric units
    """
    data_file = os.path.join(pkg_resources.resource_filename(__name__, 'data'),
                             'fitcoeffsWEB.dat')
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
    F_lmn = f1 + f2 * (1. - jj)**f3

    # angular frequency
    omega_lmn = F_lmn / MM

    # quality factor
    Q_lmn = q1 + q2 * (1. - jj)**q3

    # damping time
    tau_lmn = 2. * Q_lmn / omega_lmn

    return omega_lmn, tau_lmn


def amplitude(m1, m2, ell=2, mm=2):
    """
    calculate the amplitude of the models relative to the 22 mode.
    see equations (5) -- (8) from Gossan et al. (2012) (note there is a
    difference between the arXiv paper and the real paper)
    m1, m2 are component masses of the progenitor
    """

    from utils import m12_to_symratio

    # symmetric mass ratio
    nu = m12_to_symratio(m1, m2)

    # all modes are normalised to the A22 mode
    A22 = 0.864 * nu

    if mm == 1:
        if ell == 2:
            return 0.52 * (1 - 4*nu)**(0.71) * A22
    elif ell == 2 & mm == 2:
        return A22
    elif ell == 3 & mm == 3:
        return 0.44 * (1 - 4*nu)**(0.45) * A22
    elif ell == 4 & mm == 4:
        return (5.4 * (nu - 0.22)**2 + 0.04) * A22
    else:
        print('Unknown mode ({}, {}) specified'.format(ell, mm))


def final_mass_spin(m1, m2):
    """
    given initial total mass and mass ratio, calculate the final mass, M_f
    and dimensionless spin parameter, jj,
    using fits in Buonanno et al. (2007) -- see the caption of their Table I.
    """

    m1 = float(m1)
    m2 = float(m2)

    eta = m1 * m2 / (m1 + m2)**2
    MM = m1 + m2

    # expansion coefficients for the mass
    M_f1 = 1.
    M_f2 = (np.sqrt(8./9.) - 1.) * eta
    M_f3 = -0.498 * eta**2

    # final mass
    M_f = MM * (M_f1 + M_f2 + M_f3)

    # expansion coefficients for spin parameter
    a_f1 = np.sqrt(12.) * eta
    a_f2 = -2.9 * eta**2

    # final spin parameter -- dimensions of mass
    a_f = M_f * (a_f1 + a_f2)

    # dimensionless spin
    jj = a_f / M_f

    return M_f, jj


def hp_hx(time, m1, m2, ell, mm, iota, phi, phase):
    """
    output hplus and hcross times distance:
    both calculated using eqns. (1) and (2) from Gossan et al.
    calculates for a single ll and mm, can then sum
    iota: angle between spin axis and line-of-sight to observer
    phi: azimuth angle of BH with respect to observer
    """

    Ylm_plus = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) + (-1.)**ell * sYlm(-2, ll=ell, mm=-mm, theta=iota, phi=0)
    Ylm_cross = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) - (-1.)**ell * sYlm(-2, ll=ell, mm=-mm, theta=iota, phi=0)

    Alm = amplitude(m1, m2, ell=ell, mm=mm)

    M_f, jj = final_mass_spin(m1, m2)

    omega_lm, tau_lm = freq_damping(M_f, jj, ell=ell, mm=mm, nn=0)

    h_plus = M_f * Alm * np.exp(-time / tau_lm) * Ylm_plus * np.cos(omega_lm * time - mm * phi + phase)

    h_cross = M_f * Alm * np.exp(-time / tau_lm) * Ylm_cross * np.sin(omega_lm * time - mm * phi + phase)

    return np.real(h_plus), np.real(h_cross)


def hp_hx_template(time, omega, tau, ell, mm, iota, phi, phase):
    """
    output hplus and hcross template:
    both calculated using eqns. (1) and (2) from Gossan et al.
    but with the normalisations ignored -- i.e., A_{ell m}=1, M_f = 1
    iota: angle between spin axis and line-of-sight to observer
    phi: azimuth angle of BH with respect to observer
    """

    Ylm_plus = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) + (-1.)**ell * sYlm(-2, ll=ell, mm=-mm, theta=iota, phi=0)
    Ylm_cross = sYlm(-2, ll=ell, mm=mm, theta=iota, phi=0) - (-1.)**ell * sYlm(-2, ll=ell, mm=-mm, theta=iota, phi=0)

    h_plus = np.exp(-time / tau) * Ylm_plus * np.cos(omega * time - mm * phi + phase)

    h_cross = np.exp(-time / tau) * Ylm_cross * np.sin(omega * time - mm * phi + phase)

    return np.real(h_plus), np.real(h_cross)
