#!/bin/python
from __future__ import division, print_function
import numpy as np
import harmonics
import glob
import pkg_resources
import pandas as pd


def gamma(lm1, lm2, incs=None, theta=None, phi=None, y_lmlm_factor=None):
    """
    Coefficients mapping the spherical harmonic components of the oscillatory strain to the memory.

    Computed according to equation XYZ of Talbot et al. (2018), arXiv:18XX:ABCDE.
    Output modes with l=range(2, 20), m=m1-m2.

    Parameters
    ----------
    lm1: str
        first input spherical harmonic mode
    lm2: str
        second input spherical haromonic mode
    incs: array, optional
        observer inclination values over which to compute the final integral
    theta: array, optional
        1d array of binary inclination values, over which to compute first integral
    phi: array, optional
        1d array of binary polarisation values, over which to compute the first integral
    y_lmlm_factor: array, optional
        Array over of spherical harmonic factor evaluated on meshgrid of theta, phi

    Return
    ------
    gamma: list
        List of coefficients for output modes, l=range(2, 20), m=m1-m2
    """
    l1, m1 = int(lm1[0]), int(lm1[1:])
    l2, m2 = int(lm2[0]), int(lm2[1:])

    if incs is None:
        incs = np.linspace(0, np.pi, 500)
    pol = 0
    if theta is None:
        theta = np.linspace(0, np.pi, 250)
    if phi is None:
        phi = np.linspace(0, 2 * np.pi, 500)

    if y_lmlm_factor is None:
        s = -2
        th, ph = np.meshgrid(theta, phi)

        l1, m1 = int(lm1[0]), int(lm1[1:])
        l2, m2 = int(lm2[0]), int(lm2[1:])

        y_lmlm_factor = harmonics.sYlm(s, l1, m1, th, ph) * (-1)**(l2+m2) * harmonics.sYlm(-s, l2, -m2, th, ph)

    lambda_lm1_lm2 = np.array([lambda_lmlm(inc, pol, lm1, lm2, theta, phi, y_lmlm_factor) for inc in incs])

    sin_inc = np.sin(-incs)

    harm = {}
    for l, m in harmonics.lmax_modes(20):
        harm['{}{}'.format(l, m)] = harmonics.sYlm(-2, l, m, incs, pol)

    ells = np.arange(2, 21, 1)

    delta_m = m1 - m2
    gamma = []
    for ell in ells:
        if ell < abs(delta_m):
            gamma.append(0)
        else:
            gamma.append(np.trapz(lambda_lm1_lm2 * np.conjugate(harm['{}{}'.format(ell, delta_m)])
                                  * sin_inc, incs).real * 2 * np.pi)

    return gamma


def lambda_matrix(inc, pol, lm1, lm2, theta=None, phi=None, y_lmlm_factor=None):
    """
    Angular integral for a specific ll'mm' as given by equation XYZ of Talbot et al. (2018), arXiv:18XX:ABCDE.

    The transverse traceless part of the integral over all binary orientations is returned.

    The integral is given by:
    \int_{S^{2}} d\Omega' Y^{-2}_{\ell_1 m_1}(\Omega') \bar{Y}^{-2}_{\ell_2 m_2}(\Omega') \times \\
    \left[\frac{n_jn_k}{1-n_{l}N_{l}} \right]^{TT}

    Parameters
    ----------
    inc: float
        observer inclination
    pol: float
        oberser polarisation
    lm1: str
        first lm value format is e.g., '22'
    lm1: str
        second lm value format is e.g., '22'
    theta: array, optional
        1d array of binary inclination values, over which to integrate
    phi: array, optional
        1d array of binary polarisation values, over which to integrate
    y_lmlm_factor: array, optional
        Array over of spherical harmonic factor evaluated on meshgrid of theta, phi

    Return
    ------
    lambda_mat: array
        three by three transverse traceless matrix of the appropriate integral
    """
    if theta is None:
        theta = np.linspace(0, np.pi, 250)
    if phi is None:
        phi = np.linspace(0, 2 * np.pi, 500)

    if y_lmlm_factor is None:
        ss = -2

        th, ph = np.meshgrid(theta, phi)

        l1, m1 = int(lm1[0]), int(lm1[1:])
        l2, m2 = int(lm2[0]), int(lm2[1:])

        y_lmlm_factor = harmonics.sYlm(ss, l1, m1, th, ph) * (-1)**(l2+m2) * harmonics.sYlm(-ss, l2, -m2, th, ph)

    n = [np.outer(np.cos(phi), np.sin(theta)), np.outer(np.sin(phi), np.sin(theta)),
         np.outer(np.ones_like(phi), np.cos(theta))]
    N = [np.sin(inc) * np.cos(pol), np.sin(inc) * np.sin(pol), np.cos(inc)]
    n_dot_N = sum(n_i * N_i for n_i, N_i in zip(n, N))
    n_dot_N[n_dot_N == 1] = 0
    denominator = 1/(1-n_dot_N)

    sin_array = np.outer(phi**0, np.sin(theta))

    angle_integrals_r = np.zeros((3, 3))
    angle_integrals_i = np.zeros((3, 3))
    for j in range(3):
        for k in range(j + 1):
            # projection done here to avoid divergences
            integrand = (n[j] * n[k] - (n[j] * N[k] + n[k] * N[j]) * n_dot_N + N[j] * N[k] * n_dot_N**2)\
                        * sin_array * denominator * y_lmlm_factor
            angle_integrals_r[j, k] = np.trapz(np.trapz(np.real(integrand), theta), phi)
            angle_integrals_i[j, k] = np.trapz(np.trapz(np.imag(integrand), theta), phi)
            angle_integrals_r[k, j] = np.trapz(np.trapz(np.real(integrand), theta), phi)
            angle_integrals_i[k, j] = np.trapz(np.trapz(np.imag(integrand), theta), phi)

    proj = np.identity(3) - np.outer(N, N)
    lambda_mat = angle_integrals_r + 1j * angle_integrals_i
    lambda_mat -= proj*np.trace(lambda_mat) / 2

    return lambda_mat


def lambda_lmlm(inc, pol, lm1, lm2, theta=None, phi=None, y_lmlm_factor=None):
    """
    Angular integral for a specific ll'mm' as given by equation XYZ of Talbot et al. (2018), arXiv:18XX:ABCDE.

    The transverse traceless part of the integral over all binary orientations is returned.

    The integral is given by:
    \frac{1}{2} \int_{S^{2}} d\Omega' Y^{-2}_{\ell_1 m_1}(\Omega') \bar{Y}^{-2}_{\ell_2 m_2}(\Omega') \times \\
    \left[\frac{n_jn_k}{1-n_{l}N_{l}} \right]^{TT} (e^{+}_{jk} - i e^{\times}_{jk})

    Parameters
    ----------
    inc: float
        observer inclination
    pol: float
        oberser polarisation
    lm1: str
        first lm value format is e.g., '22'
    lm1: str
        second lm value format is e.g., '22'
    theta: array, optional
        1d array of binary inclination values, over which to integrate
    phi: array, optional
        1d array of binary polarisation values, over which to integrate
    y_lmlm_factor: array, optional
        Array over of spherical harmonic factor evaluated on meshgrid of theta, phi

    Return
    ------
    lambda_lmlm: float, complex
        lambda_plus - i lambda_cross
    """
    lambda_mat = lambda_matrix(inc, pol, lm1, lm2, theta, phi, y_lmlm_factor)

    plus, cross = omega_ij_to_omega_pol(lambda_mat, inc, pol)

    lambda_lmlm = (plus - 1j * cross) / 2

    return lambda_lmlm


def omega_ij_to_omega_pol(omega_ij, inc, pol):
    '''
    Map from strain tensor to plus and cross modes.

    We assume that only plus and cross are present.

    Parameters
    ----------
    omega_ij: array
        3x3 matrix describing strain or a proxy for strain
    inc: float
        inclination of source
    pol: float
        polarisation of source

    output:
        hp, hx - (complex) time series
    '''
    theta, phi, psi = inc, pol, 0.

    wx, wy, wz = wave_frame(theta, phi, psi)

    omega_plus = np.einsum('ij,ij->', omega_ij, plus_tensor(wx, wy, wz))
    omega_cross = np.einsum('ij,ij->', omega_ij, cross_tensor(wx, wy, wz))

    return omega_plus, omega_cross


def plus_tensor(wx, wy, wz=[0, 0, 1]):
    '''Calculate the plus polarization tensor for some basis.c.f., eq. 2 of https://arxiv.org/pdf/1710.03794.pdf'''
    e_plus = np.outer(wx, wx) - np.outer(wy, wy)
    return e_plus


def cross_tensor(wx, wy, wz=[0, 0, 1]):
    '''Calculate the cross polarization tensor for some basis.c.f., eq. 2 of https://arxiv.org/pdf/1710.03794.pdf'''
    e_cross = np.outer(wx, wy) + np.outer(wy, wx)
    return e_cross


def wave_frame(theta, phi, psi=0):
    """generate wave-frame basis from three angles, see Nishizawa et al. (2009)"""
    cth, sth = np.cos(theta), np.sin(theta)
    cph, sph = np.cos(phi), np.sin(phi)
    cps, sps = np.cos(psi), np.sin(psi)

    u = np.array([cph * cth, cth * sph, -sth])
    v = np.array([-sph, cph, 0])

    wx = -u * sps - v * cps
    wy = -u * cps + v * sps
    wz = np.cross(wx, wy)

    return wx, wy, wz


def load_gamma(data_dir=None):
    if data_dir is None:
        data_dir = pkg_resources.resource_filename(__name__, 'data')
    data_files = glob.glob('{}/gamma*.dat'.format(data_dir))
    gamma_lmlm = {}
    for file_name in data_files:
        delta_m = file_name.split('_')[-1][:-4]
        gamma_lmlm[delta_m] = pd.read_csv(file_name, sep='\t')
    return gamma_lmlm
