from typing import Tuple

import numpy as np

from ..harmonics import sYlm
from ..utils import MPC, SOLAR_MASS, combine_modes
from . import MemoryGenerator


class Approximant(MemoryGenerator):

    no_hm = ["IMRPhenomD", "IMRPhenomT", "SEOBNRv4", "IMRPhenomXAS", "TaylorF2"]
    td = ["NRSur7dq4", "SEOBNRv4PHM", "IMRPhenomTPHM", "IMRPhenomTHM"]
    fd = ["IMRPhenomXPHM", "IMRPhenomPv3HM", "IMRPhenomXHM", "IMRPhenomXP"]

    def __init__(
        self,
        name: str,
        q: float,
        total_mass: float = 60,
        spin_1: Tuple[float, float, float] = None,
        spin_2: Tuple[float, float, float] = None,
        distance: float = 400,
        times: np.ndarray = None,
        minimum_frequency: float = 20.0,
        reference_frequency: float = 20.0,
        duration: float = 4.0,
        sampling_frequency: float = 2048.0,
        l_max: int = 4,
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
            print("lalsimulation is required for the Approximant memory generator.")
            raise

        super(Approximant, self).__init__(name=name, h_lm=None, times=None, l_max=l_max)

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
        self.minimum_frequency = minimum_frequency
        self.reference_frequency = reference_frequency
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.l_max = l_max

        self.m1 = self.MTot / (1 + self.q)
        self.m2 = self.m1 * self.q
        self.m1_SI = self.m1 * SOLAR_MASS
        self.m2_SI = self.m2 * SOLAR_MASS
        self.distance_SI = self.distance * MPC

        if self._kind == "no_hm" and (
            abs(self.S1[0]) > 0
            or abs(self.S1[1]) > 0
            or abs(self.S2[0]) > 0
            or abs(self.S2[1]) > 0
        ):
            raise ValueError(
                "WARNING: Approximant decomposition works only for "
                "non-precessing waveforms."
            )

        self.h_lm = None
        self.times = None

        if times is not None:
            delta_t = times[1] - times[0]
        else:
            delta_t = None

        self.h_lm, self.times = self.time_domain_oscillatory(delta_t=delta_t)
        if times is not None:
            self.set_time_array(times)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        if name in self.no_hm:
            self._kind = "no_hm"
        elif name in self.td:
            self._kind = "td"
        elif name in self.fd:
            self._kind = "fd"
        else:
            raise ValueError(
                f"Approximant {name} is not supported. Should be one of "
                ", ".join(self.no_hm + self.td + self.fd)
            )
        self._name = name

    def time_domain_oscillatory(
        self,
        delta_t: float = None,
        modes: list = None,
        inc: float = None,
        phase: float = None,
    ) -> Tuple[dict, np.ndarray]:
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
        import lalsimulation as lalsim
        from lal import CreateDict

        if self.h_lm is None:

            params = dict(
                f_min=self.minimum_frequency,
                f_ref=self.reference_frequency,
                phiRef=0.0,
                approximant=lalsim.GetApproximantFromString(self.name),
                LALpars=None,
                m1=self.m1_SI,
                m2=self.m2_SI,
                S1x=self.S1[0],
                S1y=self.S1[1],
                S1z=self.S1[2],
                S2x=self.S2[0],
                S2y=self.S2[1],
                S2z=self.S2[2],
                distance=self.distance_SI,
                inclination=0.0,
            )

            if delta_t is None:
                params["deltaT"] = 0.1 / self.t_to_geo
            else:
                params["deltaT"] = delta_t

            if self._kind in ["td", "fd"]:
                if modes is not None:
                    WFdict = CreateDict()
                    mode_array = lalsim.SimInspiralCreateModeArray()
                    for mode in modes:
                        lalsim.SimInspiralModeArrayActivateMode(mode_array, *mode)
                    lalsim.SimInspiralWaveformParamsInsertModeArray(WFdict, modes)
                else:
                    WFdict = None
                params["LALpars"] = WFdict
            elif modes is not None:
                print(f"Modes specified for a non-HM approximant {self.name}.")

            if self._kind == "fd":
                del params["deltaT"]
                duration = self.duration
                params["deltaF"] = 1 / duration
                params["f_max"] = self.sampling_frequency / 2
                waveform_modes = lalsim.SimInspiralChooseFDModes(**params)
                times = np.arange(0, duration, 1 / self.sampling_frequency)

                h_lm = dict()
                while waveform_modes is not None:
                    mode = (waveform_modes.l, waveform_modes.m)
                    data = waveform_modes.mode.data.data[:-1]
                    h_lm[mode] = (
                        np.roll(
                            np.fft.ifft(np.roll(data, int(duration * params["f_max"]))),
                            int((duration - 1) * self.sampling_frequency),
                        )
                        * self.sampling_frequency
                    )
                    if mode == (2, 2):
                        times -= times[np.argmax(abs(h_lm[mode]))]
                    waveform_modes = waveform_modes.next
            elif self._kind == "td":
                del params["inclination"]
                params["r"] = params.pop("distance")
                params["lmax"] = self.l_max
                waveform_modes = lalsim.SimInspiralChooseTDModes(**params)
                times = np.arange(len(waveform_modes.mode.data.data)) * params["deltaT"]

                h_lm = dict()
                while waveform_modes is not None:
                    mode = (waveform_modes.l, waveform_modes.m)
                    h_lm[mode] = waveform_modes.mode.data.data
                    if mode == (2, 2):
                        times -= times[np.argmax(abs(h_lm[mode]))]
                    waveform_modes = waveform_modes.next
            elif self._kind == "no_hm":
                params["longAscNodes"] = 0.0
                params["eccentricity"] = 0.0
                params["meanPerAno"] = 0.0
                params["LALparams"] = params.pop("LALpars")
                hplus, hcross = lalsim.SimInspiralTD(**params)
                h = hplus.data.data - 1j * hcross.data.data

                h_22 = h / sYlm(-2, 2, 2, 0, 0)

                times = np.arange(len(h_22)) * params["deltaT"]
                times -= times[np.argmax(abs(h_22))]
                h_lm = {(2, 2): h_22, (2, -2): np.conjugate(h_22)}

        else:
            h_lm = self.h_lm
            times = self.times

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times
