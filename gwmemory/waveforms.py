from __future__ import print_function, division

import numpy as np
from scipy.interpolate import interp1d

import lalsimulation as lalsim
import NRSur7dq2

from . import angles, harmonics, utils, qnms


class MemoryGenerator(object):

    def __init__(self, name, h_lm, times):

        self.name = name
        self.h_lm = h_lm
        self.times = times
        self.modes = self.h_lm.keys()

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
            modes will be returned. For CBCs this is the phase at coalesence.
        gamma_lmlm: dict
            Dictionary of arrays defining the angular dependence of the
            different memory modes, default=None if None the function will
            attempt to load them.

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
            dhlm_dt[lm] = np.gradient(self.h_lm[lm], self.delta_t)

        dhlm_dt_sq = dict()
        for lm in lms:
            for lmp in lms:
                try:
                    index = (lm, lmp)
                    dhlm_dt_sq[index] = dhlm_dt[lm]*np.conjugate(dhlm_dt[lmp])
                except KeyError:
                    None

        if gamma_lmlm is None:
            gamma_lmlm = angles.load_gamma()

        # constant terms in SI units
        const = 1 / 4 / np.pi
        if self.distance is not None:
            const *= self.distance * utils.MPC / utils.CC

        dh_mem_dt_lm = dict()
        for ii, ell in enumerate(gamma_lmlm['0'].l):
            if ell > 4:
                continue
            for delta_m in gamma_lmlm.keys():
                if abs(int(delta_m)) > ell:
                    continue
                dh_mem_dt_lm[(ell, int(delta_m))] = np.sum(
                    [dhlm_dt_sq[((l1, m1), (l2, m2))] * gamma_lmlm[delta_m][
                        '{}{}{}{}'.format(l1, m1, l2, m2)][ii]
                     for (l1, m1), (l2, m2) in dhlm_dt_sq.keys()
                     if m1 - m2 == int(delta_m)], axis=0)

        h_mem_lm = {lm: const * np.cumsum(dh_mem_dt_lm[lm]) * self.delta_t
                    for lm in dh_mem_dt_lm}

        if inc is None or phase is None:
            return h_mem_lm, self.times
        else:
            return combine_modes(h_mem_lm, inc, phase), self.times

    def set_time_array(self, times):
        """
        Change the time array on which the waveform is evaluated.

        Parameters
        ----------
        times: array
            New time array for waveform to be evaluated on.
        """
        for mode in self.modes:
            interpolated_mode = interp1d(self.times, self.h_lm)
            self.h_lm[mode] = interpolated_mode[times]
        self.times = times


class Surrogate(MemoryGenerator):
    """
    Memory generator for a numerical relativity surrogate.

    Attributes
    ----------
    name: str
        Name of file to extract waveform from.
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    q: float
        Binary mass ratio
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in MPC.
    S1: array
        Spin vector of more massive black hole.
    S2: array
        Spin vector of less massive black hole.
    """

    def __init__(self, q, name='', total_mass=None, spin_1=None, spin_2=None,
                 distance=None, l_max=4, modes=None, times=None):
        """
        Initialise Surrogate MemoryGenerator

        Parameters
        ----------
        name: str
            File name to load.
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        q: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        spin_1: array
            Spin vector of more massive black hole.
        spin_2: array
            Spin vector of less massive black hole.
        times: array
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        self.name = name
        self.sur = NRSur7dq2.NRSurrogate7dq2()

        if q < 1:
            q = 1 / q
        if q > 2:
            print('WARNING: Surrogate waveform not tested for q>2.')
        self.q = q
        self.MTot = total_mass
        if spin_1 is None:
            self.S1 = np.array([0., 0., 0.])
        else:
            self.S1 = np.array(spin_1)
        if spin_2 is None:
            self.S2 = np.array([0., 0., 0.])
        else:
            self.S2 = np.array(spin_2)
        self.distance = distance
        self.LMax = l_max
        self.modes = modes

        if total_mass is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * utils.MPC / self.MTot /\
                utils.SOLAR_MASS / utils.GG * utils.CC ** 2
            self.t_to_geo = 1 / self.MTot / utils.SOLAR_MASS / utils.GG *\
                utils.CC ** 3

        self.h_lm = None
        self.times = None

        if times is not None and max(times) < 10:
            times *= self.t_to_geo

        h_lm, times = self.time_domain_oscillatory(modes=modes, times=times)

        MemoryGenerator.__init__(self, name=name, h_lm=h_lm, times=times)

    def time_domain_oscillatory(self, times=None, modes=None, inc=None,
                                phase=None):
        """
        Get the mode decomposition of the surrogate waveform.

        Calculates a BBH waveform using the surrogate models of Field et al.
        (2014), Blackman et al. (2017)
        http://journals.aps.org/prx/references/10.1103/PhysRevX.4.031006,
        https://arxiv.org/abs/1705.07089
        See https://data.black-holes.org/surrogates/index.html for more
        information.

        Parameters
        ----------
        times: np.array, optional
            Time array on which to evaluate the waveform.
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic
            modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if self.h_lm is None:
            if times is None:
                times = np.linspace(-900, 100, 10001)
            times = times / self.t_to_geo
            h_lm = self.sur(self.q, self.S1, self.S2, MTot=self.MTot,
                            distance=self.distance, t=times, LMax=self.LMax)

            available_modes = set(h_lm.keys())

            if modes is None:
                modes = available_modes

            if not set(modes).issubset(available_modes):
                print('Requested {} unavailable modes'.format(
                    ' '.join(set(modes).difference(available_modes))))
                modes = list(set(modes).union(available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            h_lm = {(ell, m): h_lm[ell, m] for ell, m in modes}

        else:
            h_lm = self.h_lm
            times = self.times

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times


class SXSNumericalRelativity(MemoryGenerator):
    """
    Memory generator for a numerical relativity waveform.

    Attributes
    ----------
    name: str
        Name of file to extract waveform from.
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in MPC.
    """

    def __init__(self, name, modes=None, extraction='OutermostExtraction.dir',
                 total_mass=None, distance=None, times=None):
        """
        Initialise SXSNumericalRelativity MemoryGenerator

        Parameters
        ----------
        name: str
            File name to load.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        extraction: str
            Extraction method, this specifies the outer object to use in the
            h5 file.
        total_mass: float
            Lab-frame total mass of the binary in solar masses.
        distace: float
            Luminosity distance to the binary in MPC.
        times: array
            Time array to evaluate the waveforms on, default is time array
            in h5 file.
        """
        self.name = name
        self.modes = modes
        self.h_lm, self.times = utils.load_sxs_waveform(
            name, modes=modes, extraction=extraction)

        self.MTot = total_mass
        self.distance = distance

        if total_mass is None or distance is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * utils.MPC / self.MTot /\
                utils.SOLAR_MASS / utils.GG * utils.CC ** 2
            self.t_to_geo = 1 / self.MTot / utils.SOLAR_MASS / utils.GG *\
                utils.CC ** 3

            for mode in self.h_lm:
                self.h_lm /= self.h_to_geo
            self.times / self.t_to_geo
            # Rezero time array to the merger time
            self.times -= self.times[np.argmax(abs(self.h_lm[(2, 2)]))]

        if times is not None:
            self.set_time_array(times)

        MemoryGenerator.__init__(self, name=name, h_lm=self.h_lm, times=times)

    def time_domain_oscillatory(self, times=None, modes=None, inc=None,
                                phase=None):
        """
        Get the mode decomposition of the numerical relativity waveform.

        Parameters
        ----------
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic
            modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if inc is None or phase is None:
            return self.h_lm, times
        else:
            return combine_modes(self.h_lm, inc, phase), times


class Approximant(MemoryGenerator):

    def __init__(self, name, q, total_mass=60, spin_1=None, spin_2=None,
                 distance=400, times=None):
        """
        Initialise Surrogate MemoryGenerator

        Parameters
        ----------
        name: str
            File name to load.
        q: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        spin_1: array
            Spin vector of more massive black hole.
        spin_2: array
            Spin vector of less massive black hole.
        times: array
            Time array to evaluate the waveforms on, default is time array
            from lalsimulation.
            FIXME
        """
        self.name = name
        if q > 1:
            q = 1 / q

        self.q = q
        self.MTot = total_mass
        if spin_1 is None:
            self.S1 = np.array([0., 0., 0.])
        else:
            self.S1 = np.array(spin_1)
        if spin_2 is None:
            self.S2 = np.array([0., 0., 0.])
        else:
            self.S2 = np.array(spin_2)
        self.distance = distance

        self.m1 = self.MTot / (1 + self.q)
        self.m2 = self.m1 * self.q
        self.m1_SI = self.m1 * utils.SOLAR_MASS
        self.m2_SI = self.m2 * utils.SOLAR_MASS
        self.distance_SI = self.distance * utils.MPC

        if abs(self.S1[0]) > 0 or abs(self.S1[1]) > 0 or abs(self.S2[0]) > 0\
                or abs(self.S2[1]) > 0:
            print('WARNING: Approximant decomposition works only for '
                  'non-precessing waveforms.')
            print('Setting spins to be aligned')
            self.S1[0], self.S1[1] = 0., 0.
            self.S2[0], self.S2[1] = 0., 0.
            print('New spins are: S1 = {}, S2 = {}'.format(self.S1, self.S2))
        else:
            self.S1 = list(self.S1)
            self.S2 = list(self.S2)
        self.available_modes = list({(2, 2), (2, -2)})

        self.h_to_geo = self.distance_SI / (self.m1_SI+self.m2_SI) / utils.GG *\
                        utils.CC ** 2
        self.t_to_geo = 1 / (self.m1_SI+self.m2_SI) / utils.GG * utils.CC ** 3

        self.h_lm = None
        self.times = None

        h_lm, times = self.time_domain_oscillatory()

        MemoryGenerator.__init__(self, name=name, h_lm=h_lm, times=times)

    def time_domain_oscillatory(self, delta_t=None, modes=None, inc=None,
                                phase=None):
        """
        Get the mode decomposition of the waveform approximant.

        Since the waveforms we consider only contain content about the
        ell=|m|=2 modes.
        We can therefore evaluate the waveform for a face-on system, where
        only the (2, 2) mode is non-zero.

        Parameters
        ----------
        delta_t: float, optional
            Time step for waveform.
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic
            modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if self.h_lm is None:
            if modes is None:
                modes = self.available_modes
            else:
                modes = modes

            if not set(modes).issubset(self.available_modes):
                print('Requested {} unavailable modes'.format(' '.join(
                    set(modes).difference(self.available_modes))))
                modes = list(set(modes).union(self.available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            fmin, fRef = 20, 20
            theta = 0.0
            phi = 0.0
            longAscNodes = 0.0
            eccentricity = 0.0
            meanPerAno = 0.0
            approx = lalsim.GetApproximantFromString(self.name)
            WFdict = None

            if delta_t is None:
                delta_t = 0.1 * (self.m1_SI + self.m2_SI) * utils.GG /\
                    utils.CC ** 3
            else:
                delta_t = delta_t

            hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
                self.m1_SI, self.m2_SI, self.S1[0], self.S1[1], self.S1[2],
                self.S2[0], self.S2[1], self.S2[2], self.distance_SI, theta,
                phi, longAscNodes, eccentricity, meanPerAno, delta_t, fmin,
                fRef, WFdict, approx)

            h = hplus.data.data - 1j * hcross.data.data

            h_22 = h / harmonics.sYlm(-2, 2, 2, theta, phi)

            times = np.linspace(0, delta_t * len(h), len(h))
            times -= times[np.argmax(abs(h_22))]

            h_lm = {(2, 2): h_22, (2, -2): np.conjugate(h_22)}

        else:
            h_lm = self.h_lm
            times = self.times

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times


class MWM(MemoryGenerator):

    def __init__(self, q, total_mass=60, distance=400, name='MWM', times=None):
        MemoryGenerator.__init__(self, name=name, h_lm=dict(), times=times)
        self.name = name
        if q > 1:
            q = 1 / q
        self.q = q
        self.MTot = total_mass
        self.distance = distance
        self.m1 = self.MTot / (1 + self.q)
        self.m2 = self.m1 * self.q

        self.h_to_geo = self.distance * utils.MPC / (self.m1 + self.m2) /\
            utils.SOLAR_MASS / utils.GG * utils.CC ** 2
        self.t_to_geo = 1 / (self.m1+self.m2) / utils.SOLAR_MASS / utils.GG *\
            utils.CC ** 3

        if times is None:
            times = np.linspace(-900, 100, 10001) / self.t_to_geo
        self.times = times

    def time_domain_memory(self, inc, phase, times=None, rm=3):
        """
        Calculates the plus and cross polarisations for the
        minimal waveform model memory waveform:
        eqns (5) and (9) from Favata (2009. ApJL 159)

        TODO: Impement spherical harmonic decomposition?

        Parameters
        ----------
        inc: float
            Binary inclination angle
        phase: float
            Binary phase at coalscence
        times: array, optional
            Time array on which the memory is calculated
        rm: float, optional
            Radius at which the matching occurs in solar masses

        Returns
        -------
        h_mem: dict
            Plus and cross polarisations of the memory waveform
        times: np.array
            Time array on which the memory is calculated

        Paul Lasky
        """
        if times is None:
            times = self.times

        time_geo = utils.time_s_to_geo(times)  # units: metres

        m1_geo = utils.m_sol_to_geo(self.m1)   # units: metres
        m2_geo = utils.m_sol_to_geo(self.m2)   # units: metres

        dist_geo = utils.dist_Mpc_to_geo(self.distance)  # units: metres

        # total mass
        MM = m1_geo + m2_geo

        # symmetric mass ratio
        eta = utils.m12_to_symratio(m1_geo, m2_geo)

        # this is the orbital separation at the matching radius --
        # see Favata (2009) before eqn (8).
        # the default value for this is given as rm = 3 MM.
        rm *= MM

        # calculate dimensionless mass and spin of the final black hole
        # from the Buonanno et al. (2007) fits
        Mf_geo, jj = qnms.final_mass_spin(m1_geo, m2_geo)

        # calculate the QNM frequencies and damping times
        # from the fits in Table VIII of Berti et al. (2006)
        omega220, tau220 = qnms.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=0)
        omega221, tau221 = qnms.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=1)
        omega222, tau222 = qnms.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=2)

        sigma220 = 1j * omega220 + 1 / tau220
        sigma221 = 1j * omega221 + 1 / tau221
        sigma222 = 1j * omega222 + 1 / tau222

        # set the matching time to be at t = 0
        tm = 0
        TT = time_geo - tm

        # some quantity defined after equation (7) of Favata
        trr = 5 * MM * rm**4 / (256 * eta * MM**4)

        # calculate the A_{ell m n} matching coefficients.  Note that
        # I've solved a matrix equation that solved for the three coefficients
        # from three equations
        xi = 2 * np.sqrt(2 * np.pi / 5) * eta * MM * rm**2
        chi = -2 * 1j * np.sqrt(MM / rm**3)

        A220 = xi * (sigma221 * sigma222 * chi**2 + sigma221 * chi**3 +
                     sigma222 * chi**3 + chi**4)\
            / ((sigma220 - sigma221) * (sigma220 - sigma222))
        A221 = xi * (sigma220 * sigma222 * chi**2 + sigma220 * chi**3 +
                     sigma222 * chi**3 + chi**4)\
            / ((sigma221 - sigma220) * (sigma221 - sigma222))
        A222 = xi * (sigma220 * sigma221 * chi**2 + sigma220 * chi**3 +
                     sigma221 * chi**3 + chi**4)\
            / ((sigma221 - sigma222) * (sigma220 - sigma222))

        # Calculate the coefficients in the summed term of equation (9)
        # from Favata (2009) this is a double sum, with each variable going
        # from n = 0 to 2; therefore 9 terms
        coeffSum00 = sigma220 * np.conj(sigma220) * A220 * np.conj(A220) /\
            (sigma220 + np.conj(sigma220))
        coeffSum01 = sigma220 * np.conj(sigma221) * A220 * np.conj(A221) /\
            (sigma220 + np.conj(sigma221))
        coeffSum02 = sigma220 * np.conj(sigma222) * A220 * np.conj(A222) /\
            (sigma220 + np.conj(sigma222))

        coeffSum10 = sigma221 * np.conj(sigma220) * A221 * np.conj(A220) /\
            (sigma221 + np.conj(sigma220))
        coeffSum11 = sigma221 * np.conj(sigma221) * A221 * np.conj(A221) /\
            (sigma221 + np.conj(sigma221))
        coeffSum12 = sigma221 * np.conj(sigma222) * A221 * np.conj(A222) /\
            (sigma221 + np.conj(sigma222))

        coeffSum20 = sigma222 * np.conj(sigma220) * A222 * np.conj(A220) /\
            (sigma222 + np.conj(sigma220))
        coeffSum21 = sigma222 * np.conj(sigma221) * A222 * np.conj(A221) /\
            (sigma222 + np.conj(sigma221))
        coeffSum22 = sigma222 * np.conj(sigma222) * A222 * np.conj(A222) /\
            (sigma222 + np.conj(sigma222))

        # radial separation
        rr = rm * (1 - TT / trr)**(1 / 4)

        # set up strain
        h_MWM = np.zeros(len(TT))

        # calculate strain for TT < 0.
        h_MWM[TT <= 0] = 8 * np.pi * MM / rr[TT <= 0]

        # calculate strain for TT > 0.
        term00 = coeffSum00 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma220 + np.conj(sigma220))))
        term01 = coeffSum01 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma220 + np.conj(sigma221))))
        term02 = coeffSum02 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma220 + np.conj(sigma222))))

        term10 = coeffSum10 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma221 + np.conj(sigma220))))
        term11 = coeffSum11 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma221 + np.conj(sigma221))))
        term12 = coeffSum12 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma221 + np.conj(sigma222))))

        term20 = coeffSum20 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma222 + np.conj(sigma220))))
        term21 = coeffSum21 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma222 + np.conj(sigma221))))
        term22 = coeffSum22 * (1 - np.exp(-TT[TT > 0] *
                                          (sigma222 + np.conj(sigma222))))

        sum_terms = np.real(term00 + term01 + term02 +
                            term10 + term11 + term12 +
                            term20 + term21 + term22)

        h_MWM[TT > 0] = 8 * np.pi * MM / rm + sum_terms / (eta * MM)

        # calculate the plus polarisation of GWs: eqn. (5) from Favata (2009)
        sT = np.sin(inc)
        cT = np.cos(inc)

        h_plus_coeff = 0.77 * eta * MM / (384 * np.pi) * sT**2 * (17 + cT**2) /\
            dist_geo
        h_mem = dict(plus=h_plus_coeff * h_MWM, cross=np.zeros_like(h_MWM))

        return h_mem, times


def combine_modes(h_lm, inc, phase):
    """
    Calculate the plus and cross polarisations of the waveform from the
    spherical harmonic decomposition.
    """
    total = sum([h_lm[(l, m)] * harmonics.sYlm(-2, l, m, inc, phase)
                 for l, m in h_lm])
    h_plus_cross = dict(plus=total.real, cross=-total.imag)
    return h_plus_cross
