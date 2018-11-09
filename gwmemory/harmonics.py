# ----------------------------------------------------------
#
# This module computes spin-weighted spherical harmonics.
#
# Released under the MIT License.
# (C) Christian Reisswig 2009-2011
# Modified Colm Talbot 2018
#
# ----------------------------------------------------------

import numpy as np


def fac(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result


# coefficient function
def Cslm(s, l, m):
    return np.sqrt(l*l * (4.0*l*l - 1.0) / ((l*l - m*m) * (l*l - s*s)))


# recursion function
def s_lambda_lm(s, l, m, x):
    Pm = pow(-0.5, m)
    if m != s:
        Pm = Pm * pow(1.0+x, (m-s)*1.0/2)
    if m != -s:
        Pm = Pm * pow(1.0-x, (m+s)*1.0/2)
    Pm = Pm * np.sqrt(fac(2*m + 1) * 1.0 / (4.0*np.pi * fac(m+s) * fac(m-s)))
    if l == m:
        return Pm
    Pm1 = (x + s*1.0/(m+1)) * Cslm(s, m+1, m) * Pm
    if l == m+1:
        return Pm1
    else:
        for n in range(m+2, l+1):
            Pn = (x + s*m * 1.0 / (n * (n-1.0))) * Cslm(s, n, m) * Pm1 - Cslm(s, n, m) * 1.0 / Cslm(s, n-1, m) * Pm
            Pm = Pm1
            Pm1 = Pn
        return Pn


def sYlm(ss, ll, mm, theta, phi):
    """Calculate spin-weighted harmonic"""
    Pm = 1.0
    l = ll
    m = mm
    s = ss
    if l < 0:
        return 0
    if abs(m) > l or l < abs(s):
        return 0
    if abs(mm) < abs(ss):
        s = mm
        m = ss
    if (m+s) % 2:
        Pm = -Pm
    if m < 0:
        s = -s
        m = -m
    if (m+s) % 2:
        Pm = -Pm
    result = Pm * s_lambda_lm(s, l, m, np.cos(theta))
    return result * np.cos(mm*phi) + result * np.sin(mm*phi) * 1j


def lmax_modes(lmax):
    """Compute all (l, m) pairs with 2<=l<=lmax"""
    return [(l, m) for l in range(2, lmax+1) for m in range(-l, l+1)]
