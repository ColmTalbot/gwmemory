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
