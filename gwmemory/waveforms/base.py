import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

from ..angles import load_gamma
from ..utils import CC, MPC, combine_modes


class MemoryGenerator(object):
    def __init__(self, name, h_lm, times):

        self.name = name
        self.h_lm = h_lm
        self.times = times
        self.modes = self.h_lm.keys()

    @property
    def distance(self):
        return getattr(self, "_distance", None)

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @property
    def delta_t(self):
        return self.times[1] - self.times[0]

    def time_domain_memory(self, inc=None, phase=None, gamma_lmlm=None):
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

        lms = self.modes

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

        if gamma_lmlm is None:
            gamma_lmlm = load_gamma()

        # constant terms in SI units
        const = 1 / 4 / np.pi
        if self.distance is not None:
            const *= self.distance * MPC / CC

        dh_mem_dt_lm = dict()
        for ii, ell in enumerate(gamma_lmlm["0"].l):
            if ell > 4:
                continue
            for delta_m in gamma_lmlm.keys():
                if abs(int(delta_m)) > ell:
                    continue
                dh_mem_dt_lm[(ell, int(delta_m))] = np.sum(
                    [
                        dhlm_dt_sq[((l1, m1), (l2, m2))]
                        * gamma_lmlm[delta_m][f"{l1}{m1}{l2}{m2}"][ii]
                        for (l1, m1), (l2, m2) in dhlm_dt_sq.keys()
                        if m1 - m2 == int(delta_m)
                        and f"{l1}{m1}{l2}{m2}" in gamma_lmlm[delta_m]
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
            output[mode] = interp1d(self.times, h_lm[mode])(times)
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
