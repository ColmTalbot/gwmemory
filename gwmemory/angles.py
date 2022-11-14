#!/usr/bin/python3
from functools import lru_cache
from typing import Tuple

import numpy as np
from sympy.physics.wigner import wigner_3j

from . import harmonics


@lru_cache
def memory_correction(ell: int, ss: int = 0) -> float:
    """
    Correction to the Gamma function for the operator in Eq. (12) of
    arXiv:2011.01309

    Parameters
    ----------
    ell: int
        degree of the spherical harmonic
    ss: int
        spin-weight of the waveform being adjusted, ss=0 for out purpose

    Returns
    -------
    int: the correction

    """
    if ell < 2:
        return 0
    return (
        ((ell - (ss - 1)) * (ell + ss) * (ell - (ss - 2)) * (ell + (ss - 1))) ** 0.5
        * 4
        / ((ell + 2) * (ell + 1) * ell * (ell - 1))
    )


@lru_cache()
def analytic_gamma(lm1: Tuple[int, int], lm2: Tuple[int, int], ell: int) -> float:
    """
    Analytic function to compute gamma_lmlm_l Eq. (8) of arXiv:1807.0090

    The primary component is taken from https://github.com/moble/spherical/blob/c3fe00ab6d79732fe1cbc6d56574ea94702d89ae/spherical/multiplication.py.

    Parameters
    ----------
    lm1: tuple
        tuple of first spherical harmonic mode
    lm2: tuple
        tuple of second spherical harmonic mode
    ell: int
        The degree of the output spherical harmonic

    Returns
    -------
    float: the gamma coefficient

    """
    ell1, m1 = lm1
    ell2, m2 = lm2
    s1, s2, s3 = -2, 2, 0
    m2 = -m2
    m3 = m1 + m2
    return (
        (-1.0) ** (ell1 + ell2 + ell + m3 + m2)
        * (2 * ell1 + 1) ** 0.5
        * (2 * ell2 + 1) ** 0.5
        * (2 * ell + 1) ** 0.5
        * float(
            wigner_3j(ell1, ell2, ell, s1, s2, -s3)
            * wigner_3j(ell1, ell2, ell, m1, m2, -m3)
        )
        * np.pi**0.5
        / 2
        * memory_correction(ell)
    )


def gamma(
    lm1: str,
    lm2: str,
    incs: np.ndarray = None,
    theta: np.ndarray = None,
    phi: np.ndarray = None,
    y_lmlm_factor: np.ndarray = None,
) -> list:
    """
    Coefficients mapping the spherical harmonic components of the oscillatory
    strain to the memory.

    Computed according to equation 8 of Talbot et al. (2018), arXiv:1807.00990.
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
        1d array of binary inclination values, over which to compute first
        integral
    phi: array, optional
        1d array of binary polarisation values, over which to compute the first
        integral
    y_lmlm_factor: array, optional
        Array over of spherical harmonic factor evaluated on meshgrid of theta,
        phi

    Returns
    -------
    gammas: list
        List of coefficients for output modes, l=range(2, 20), m=m1-m2

    Notes
    -----
    I recommend using :code:`analytic_gamma` instead, it is much more precise.
    """
    l1, m1 = int(lm1[0]), int(lm1[1:])
    l2, m2 = int(lm2[0]), int(lm2[1:])

    if incs is None:
        incs = np.linspace(0, np.pi, 500)
    phase = 0

    if y_lmlm_factor is None:
        y_lmlm_factor, theta, phi = ylmlm_factor(theta=theta, phi=phi, lm1=lm1, lm2=lm2)

    lambda_lm1_lm2 = np.array(
        [lambda_lmlm(inc, phase, lm1, lm2, theta, phi, y_lmlm_factor) for inc in incs]
    )

    sin_inc = -np.sin(incs)

    harm = {}
    for l, m in harmonics.lmax_modes(20):
        harm[f"{l}{m}"] = harmonics.sYlm(-2, l, m, incs, phase)

    ells = np.arange(2, 21, 1)

    delta_m = m1 - m2
    gammas = []
    for ell in ells:
        if ell < abs(delta_m):
            gammas.append(0)
        else:
            gammas.append(
                np.real(
                    2
                    * np.pi
                    * np.trapz(
                        lambda_lm1_lm2
                        * np.conjugate(harm[f"{ell}{-delta_m}"])
                        * sin_inc,
                        incs,
                    )
                )
            )

    return gammas


def ylmlm_factor(theta: np.ndarray, phi: np.ndarray, lm1: str, lm2: str) -> np.ndarray:
    if theta is None:
        theta = np.linspace(0, np.pi, 250)
    if phi is None:
        phi = np.linspace(0, 2 * np.pi, 500)

    ss = -2

    th, ph = np.meshgrid(theta, phi)

    l1, m1 = int(lm1[0]), int(lm1[1:])
    l2, m2 = int(lm2[0]), int(lm2[1:])

    y_lmlm_factor = (
        harmonics.sYlm(ss, l1, m1, th, ph)
        * np.conjugate(harmonics.sYlm(ss, l2, m2, th, ph))
        * (-1) ** (l1 + l2)
    )
    return y_lmlm_factor, theta, phi


def lambda_matrix(
    inc: float,
    phase: float,
    lm1: str,
    lm2: str,
    theta: np.ndarray = None,
    phi: np.ndarray = None,
    y_lmlm_factor: np.ndarray = None,
) -> np.ndarray:
    r"""
    Angular integral for a specific ll'mm' as given by equation 7 of Talbot
    et al. (2018), arXiv:1807.00990.

    The transverse traceless part of the integral over all binary orientations
    is returned.

    The integral is given by:
    \int_{S^{2}} d\Omega' Y^{-2}_{\ell_1 m_1}(\Omega')
    \bar{Y}^{-2}_{\ell_2 m_2}(\Omega') \times \\
    \left[\frac{n_jn_k}{1-n_{l}N_{l}} \right]^{TT}

    Parameters
    ----------
    inc: float
        binary inclination
    phase: float
        binary phase at coalescence
    lm1: str
        first lm value format is e.g., '22'
    lm2: str
        second lm value format is e.g., '22'
    theta: array, optional
        1d array of binary inclination values, over which to integrate
    phi: array, optional
        1d array of binary polarisation values, over which to integrate
    y_lmlm_factor: array, optional
        Array over of spherical harmonic factor evaluated on meshgrid of
        theta, phi

    Returns
    -------
    lambda_mat: array
        three by three transverse traceless matrix of the appropriate integral
    """
    if y_lmlm_factor is None:
        y_lmlm_factor, theta, phi = ylmlm_factor(theta=theta, phi=phi, lm1=lm1, lm2=lm2)

    n = np.array(
        [
            np.outer(np.cos(phi), np.sin(theta)),
            np.outer(np.sin(phi), np.sin(theta)),
            np.outer(np.ones_like(phi), np.cos(theta)),
        ]
    )
    line_of_sight = np.array(
        [np.sin(inc) * np.cos(phase), np.sin(inc) * np.sin(phase), np.cos(inc)]
    )
    n_dot_line_of_sight = sum(n_i * N_i for n_i, N_i in zip(n, line_of_sight))
    n_dot_line_of_sight[n_dot_line_of_sight == 1] = 0
    denominator = 1 / (1 - n_dot_line_of_sight)

    sin_array = np.outer(phi**0, np.sin(theta))

    angle_integrals_r = np.zeros((3, 3))
    angle_integrals_i = np.zeros((3, 3))
    for j in range(3):
        for k in range(j + 1):
            # projection done here to avoid divergences
            integrand = (
                sin_array
                * denominator
                * y_lmlm_factor
                * (
                    n[j] * n[k]
                    - (n[j] * line_of_sight[k] + n[k] * line_of_sight[j])
                    * n_dot_line_of_sight
                    + line_of_sight[j] * line_of_sight[k] * n_dot_line_of_sight**2
                )
            )
            angle_integrals_r[j, k] = np.trapz(np.trapz(np.real(integrand), theta), phi)
            angle_integrals_i[j, k] = np.trapz(np.trapz(np.imag(integrand), theta), phi)
            angle_integrals_r[k, j] = np.trapz(np.trapz(np.real(integrand), theta), phi)
            angle_integrals_i[k, j] = np.trapz(np.trapz(np.imag(integrand), theta), phi)

    proj = np.identity(3) - np.outer(line_of_sight, line_of_sight)
    lambda_mat = angle_integrals_r + 1j * angle_integrals_i
    lambda_mat -= proj * np.trace(lambda_mat) / 2

    return lambda_mat


def lambda_lmlm(
    inc: float,
    phase: float,
    lm1: str,
    lm2: str,
    theta: np.ndarray = None,
    phi: np.ndarray = None,
    y_lmlm_factor: np.ndarray = None,
) -> complex:
    r"""
    Angular integral for a specific ll'mm' as given by equation 7 of Talbot
    et al. (2018), arXiv:1807.00990.

    The transverse traceless part of the integral over all binary orientations
    is returned.

    The integral is given by:
    \frac{1}{2} \int_{S^{2}} d\Omega' Y^{-2}_{\ell_1 m_1}(\Omega')
    \bar{Y}^{-2}_{\ell_2 m_2}(\Omega') \times \\
    \left[\frac{n_jn_k}{1-n_{l}N_{l}} \right]^{TT} (e^{+}_{jk} -
    i e^{\times}_{jk})

    Parameters
    ----------
    inc: float
        binary inclination
    phase: float
        binary phase at coalescence
    lm1: str
        first lm value format is e.g., '22'
    lm2: str
        second lm value format is e.g., '22'
    theta: array, optional
        1d array of binary inclination values, over which to integrate
    phi: array, optional
        1d array of binary polarisation values, over which to integrate
    y_lmlm_factor: array, optional
        Array over of spherical harmonic factor evaluated on meshgrid of
        theta, phi

    Returns
    -------
    lambda_lmlm: float, complex
        lambda_plus - i lambda_cross
    """
    lambda_mat = lambda_matrix(inc, phase, lm1, lm2, theta, phi, y_lmlm_factor)

    plus, cross = omega_ij_to_omega_pol(lambda_mat, inc, phase)

    lambda_lmlm = (plus - 1j * cross) / 2

    return lambda_lmlm


def omega_ij_to_omega_pol(
    omega_ij: np.ndarray, inc: float, phase: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map from strain tensor to plus and cross modes.

    We assume that only plus and cross are present.

    Parameters
    ----------
    omega_ij: array
        3x3 matrix describing strain or a proxy for strain
    inc: float
        inclination of source
    phase: float
        phase at coalescence of source

    Returns
    -------
    hp: float
        Magnitude of plus mode.
    hx: float
        Magnitude of cross mode.
    """
    psi = 0.0

    wx, wy = wave_frame(inc, phase, psi)

    omega_plus = np.einsum("ij,ij->", omega_ij, plus_tensor(wx, wy))
    omega_cross = np.einsum("ij,ij->", omega_ij, cross_tensor(wx, wy))

    return omega_plus, omega_cross


def plus_tensor(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    Calculate the plus polarization tensor for some basis.
    c.f., eq. 2 of https://arxiv.org/pdf/1710.03794.pdf
    """
    e_plus = np.outer(wx, wx) - np.outer(wy, wy)
    return e_plus


def cross_tensor(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    Calculate the cross polarization tensor for some basis.
    c.f., eq. 2 of https://arxiv.org/pdf/1710.03794.pdf
    """
    e_cross = np.outer(wx, wy) + np.outer(wy, wx)
    return e_cross


def wave_frame(
    theta: float, phi: float, psi: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate wave-frame basis from three angles, see Nishizawa et al. (2009)
    """
    cth, sth = np.cos(theta), np.sin(theta)
    cph, sph = np.cos(phi), np.sin(phi)
    cps, sps = np.cos(psi), np.sin(psi)

    u = np.array([cph * cth, cth * sph, -sth])
    v = np.array([-sph, cph, 0])

    wx = -u * sps - v * cps
    wy = -u * cps + v * sps

    return wx, wy
