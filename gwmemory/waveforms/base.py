import warnings
from typing import Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ..angles import analytic_gamma
from ..harmonics import lmax_modes
from ..utils import CC, GG, MPC, SOLAR_MASS, combine_modes


class MemoryGenerator(object):
    def __init__(self, name: str, h_lm: dict, times: np.ndarray, l_max: int = 4):

        self.name = name
        self.h_lm = h_lm
        self.times = times
        self.l_max = l_max

    @property
    def h_to_geo(self):
        if self.MTot is None or self.distance is None:
            return 1
        else:
            return self.distance * MPC / self.MTot / SOLAR_MASS / GG * CC**2

    @property
    def t_to_geo(self):
        if self.MTot is None or self.distance is None:
            return 1
        else:
            return 1 / self.MTot / SOLAR_MASS / GG * CC**3

    @property
    def modes(self):
        return list(self.h_lm.keys())

    @property
    def distance(self):
        return getattr(self, "_distance", None)

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @property
    def delta_t(self):
        return self.times[1] - self.times[0]

    def time_domain_memory(
        self,
        inc: float = None,
        phase: float = None,
        modes: list = None,
        gamma_lmlm: dict = None,
    ) -> Tuple[dict, np.ndarray]:
        """
        Calculate the spherical harmonic decomposition of the nonlinear
        memory from a dictionary of spherical mode time series

        Parameters
        ----------
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Reference phase of the source, if None, the spherical harmonic
            modes will be returned. For CBCs this is the phase at coalescence.
        modes: list
            The modes to consider when computing the memory. By default all
            available modes will be used.
        gamma_lmlm: dict, deprecated
            Dictionary of arrays defining the angular dependence of the
            different memory modes, these are now computed/cached on the fly.

        Return
        ------
        h_mem_lm: dict
            Time series of the spherical harmonic decomposed memory waveform.
        times: array
            Time series on which memory is evaluated.
        """

        if modes is None:
            lms = self.modes
        else:
            lms = modes

        dhlm_dt = dict()
        for lm in lms:
            dhlm_dt[lm] = np.gradient(self.h_lm[lm], self.times)

        dhlm_dt_sq = dict()
        for lm in lms:
            for lmp in lms:
                try:
                    index = (lm, lmp)
                    dhlm_dt_sq[index] = dhlm_dt[lm] * np.conjugate(dhlm_dt[lmp])
                except KeyError:
                    pass

        if gamma_lmlm is not None:
            warnings.warn(f"The gamma_lmlm argument is deprecated and will be removed.")

        # constant terms in SI units
        const = 1 / 4 / np.pi
        if self.distance is not None:
            const *= self.distance * MPC / CC

        dh_mem_dt_lm = dict()

        modes = lmax_modes(self.l_max)
        for ell, delta_m in modes:
            dh_mem_dt_lm[(ell, int(delta_m))] = np.sum(
                [
                    dhlm_dt_sq[(lm1, lm2)] * analytic_gamma(lm1, lm2, ell)
                    for lm1, lm2 in dhlm_dt_sq.keys()
                    if delta_m == lm1[1] - lm2[1]
                ],
                axis=0,
            ) * np.ones(len(self.times), dtype=complex)
        h_mem_lm = {
            lm: const * cumulative_trapezoid(dh_mem_dt_lm[lm], self.times, initial=0)
            for lm in dh_mem_dt_lm
        }

        if inc is None or phase is None:
            return h_mem_lm, self.times
        else:
            return combine_modes(h_mem_lm, inc, phase), self.times

    def apply_time_array(self, times, h_lm=None):
        if h_lm is None:
            h_lm = self.h_lm
        output = dict()
        for mode in h_lm:
            output[mode] = interp1d(
                self.times,
                h_lm[mode],
                fill_value=0,
                bounds_error=False,
            )(times)
        return output

    def set_time_array(self, times):
        """
        Change the time array on which the waveform is evaluated.

        Parameters
        ----------
        times: array
            New time array for waveform to be evaluated on.
        """
        self.h_lm = self.apply_time_array(times)
        self.times = times
