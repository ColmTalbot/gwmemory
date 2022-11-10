import numpy as np

from ..harmonics import sYlm
from ..utils import combine_modes, CC, GG, MPC, SOLAR_MASS
from . import MemoryGenerator


class Approximant(MemoryGenerator):
    def __init__(
        self, name, q, total_mass=60, spin_1=None, spin_2=None, distance=400, times=None
    ):
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
        spin_1: array-like
            Spin vector of more massive black hole.
        spin_2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is time array
            from lalsimulation.
        """
        try:
            import lalsimulation  # noqa
        except ModuleNotFoundError:
            print("lalsuite is required for the Approximant memory generator.")
            raise
        self.name = name
        if q > 1:
            q = 1 / q

        self.q = q
        self.MTot = total_mass
        if spin_1 is None:
            self.S1 = np.array([0.0, 0.0, 0.0])
        else:
            self.S1 = np.array(spin_1).astype(float)
        if spin_2 is None:
            self.S2 = np.array([0.0, 0.0, 0.0])
        else:
            self.S2 = np.array(spin_2).astype(float)
        self.distance = distance

        self.m1 = self.MTot / (1 + self.q)
        self.m2 = self.m1 * self.q
        self.m1_SI = self.m1 * SOLAR_MASS
        self.m2_SI = self.m2 * SOLAR_MASS
        self.distance_SI = self.distance * MPC

        if (
            abs(self.S1[0]) > 0
            or abs(self.S1[1]) > 0
            or abs(self.S2[0]) > 0
            or abs(self.S2[1]) > 0
        ):
            print(
                "WARNING: Approximant decomposition works only for "
                "non-precessing waveforms."
            )
            print("Setting spins to be aligned")
            self.S1[0], self.S1[1] = 0.0, 0.0
            self.S2[0], self.S2[1] = 0.0, 0.0
            print("New spins are: S1 = {}, S2 = {}".format(self.S1, self.S2))
        else:
            self.S1 = list(self.S1)
            self.S2 = list(self.S2)
        self.available_modes = list({(2, 2), (2, -2)})

        self.h_to_geo = self.distance_SI / (self.m1_SI + self.m2_SI) / GG * CC ** 2
        self.t_to_geo = 1 / (self.m1_SI + self.m2_SI) / GG * CC ** 3

        self.h_lm = None
        self.times = None

        h_lm, _times = self.time_domain_oscillatory()
        MemoryGenerator.__init__(self, name=name, h_lm=h_lm, times=_times)
        if times is not None:
            self.set_time_array(times)

    def time_domain_oscillatory(self, delta_t=None, modes=None, inc=None, phase=None):
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
        from lalsimulation import GetApproximantFromString, SimInspiralChooseTDWaveform

        if self.h_lm is None:
            if modes is None:
                modes = self.available_modes
            else:
                modes = modes

            if not set(modes).issubset(self.available_modes):
                print(
                    "Requested {} unavailable modes".format(
                        " ".join(set(modes).difference(self.available_modes))
                    )
                )
                modes = list(set(modes).union(self.available_modes))
                print("Using modes {}".format(" ".join(modes)))

            fmin, fRef = 20, 20
            theta = 0.0
            phi = 0.0
            longAscNodes = 0.0
            eccentricity = 0.0
            meanPerAno = 0.0
            approx = GetApproximantFromString(self.name)
            WFdict = None

            if delta_t is None:
                delta_t = 0.1 * (self.m1_SI + self.m2_SI) * GG / CC ** 3
            else:
                delta_t = delta_t

            hplus, hcross = SimInspiralChooseTDWaveform(
                self.m1_SI,
                self.m2_SI,
                self.S1[0],
                self.S1[1],
                self.S1[2],
                self.S2[0],
                self.S2[1],
                self.S2[2],
                self.distance_SI,
                theta,
                phi,
                longAscNodes,
                eccentricity,
                meanPerAno,
                delta_t,
                fmin,
                fRef,
                WFdict,
                approx,
            )

            h = hplus.data.data - 1j * hcross.data.data

            h_22 = h / sYlm(-2, 2, 2, theta, phi)

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
