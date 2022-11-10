import numpy as np

from ..utils import combine_modes, CC, GG, MPC, SOLAR_MASS
from . import MemoryGenerator


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

    def __init__(
        self,
        q,
        name="nrsur7dq2",
        total_mass=None,
        spin_1=None,
        spin_2=None,
        distance=None,
        l_max=4,
        modes=None,
        times=None,
        minimum_frequency=0,
        sampling_frequency=None,
    ):
        """
        Initialise Surrogate MemoryGenerator

        Parameters
        ----------
        name: str
            Name of the surrogate, default=NRSur7dq2.
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
        spin_1: array-like
            Spin vector of more massive black hole.
        spin_2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        self.name = name

        if name.lower() == "nrsur7dq2":
            try:
                from NRSur7dq2 import NRSurrogate7dq2
            except ModuleNotFoundError:
                print("nrsur7sq2 is required for the Surrogate memory generator.")
                print("$ python -m pip install nrsur7dq2")
                raise

            self.sur = NRSurrogate7dq2()

            if q < 1:
                q = 1 / q
            if q > 2:
                print("WARNING: Surrogate waveform not tested for q>2.")
            self.q = q
            self.MTot = total_mass
            if spin_1 is None:
                self.S1 = np.array([0.0, 0.0, 0.0])
            else:
                self.S1 = np.array(spin_1)
            if spin_2 is None:
                self.S2 = np.array([0.0, 0.0, 0.0])
            else:
                self.S2 = np.array(spin_2)

        else:
            try:
                import gwsurrogate
            except ModuleNotFoundError:
                print("gwsurrogate is required for the Surrogate memory generator.")
                print("$ conda install -c conda-forge gwsurrogate")
                raise
            try:
                self.sur = gwsurrogate.LoadSurrogate(name)
            except ValueError:
                raise ValueError(f"Surrogate model {name} not in {gwsurrogate.SURROGATE_CLASSES}")
            if q < 1:
                q = 1 / q
            self.q = q
            self.MTot = total_mass
            if spin_1 is None:
                self.S1 = np.array([0.0, 0.0, 0.0])
            else:
                self.S1 = spin_1
            if spin_2 is None:
                self.S2 = np.array([0.0, 0.0, 0.0])
            else:
                self.S2 = spin_2
            self.minimum_frequency = minimum_frequency
            self.sampling_frequency = sampling_frequency
        self.distance = distance
        self.LMax = l_max
        self.modes = modes

        if total_mass is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * MPC / self.MTot / SOLAR_MASS / GG * CC ** 2
            self.t_to_geo = 1 / self.MTot / SOLAR_MASS / GG * CC ** 3

        self.h_lm = None
        self.times = times

        if times is not None and max(times) < 10:
            times = times * self.t_to_geo

        h_lm, times = self.time_domain_oscillatory(modes=modes, times=times)

        MemoryGenerator.__init__(self, name=name, h_lm=h_lm, times=times)

    def time_domain_oscillatory(self, times=None, modes=None, inc=None, phase=None):
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
            if self.name.lower() == "nrsur7dq2":
                if times is None:
                    times = np.linspace(-900, 100, 10001)
                times = times / self.t_to_geo
                h_lm = self.sur(
                    self.q,
                    self.S1,
                    self.S2,
                    MTot=self.MTot,
                    distance=self.distance,
                    t=times,
                    LMax=self.LMax,
                )
            else:
                if self.MTot is None:
                    units = "dimensionless"
                else:
                    units = "mks"
                if self.sampling_frequency is not None:
                    delta_t = 1 / self.sampling_frequency
                else:
                    delta_t = None
                times, h_lm, _ = self.sur(
                    q=self.q,
                    chiA0=self.S1,
                    chiB0=self.S2,
                    M=self.MTot,
                    dist_mpc=self.distance,
                    dt=delta_t,
                    f_low=self.minimum_frequency,
                    mode_list=self.modes,
                    units=units,
                )
                if (5, 5) in h_lm:
                    del h_lm[(5, 5)]
                old_keys = [(ll, mm) for ll, mm in h_lm.keys()]
                for ll, mm in old_keys:
                    if mm > 0 and (ll, -mm) not in h_lm:
                        h_lm[(ll, -mm)] = (-1) ** ll * np.conj(h_lm[(ll, mm)])

            available_modes = set(h_lm.keys())

            if modes is None:
                modes = available_modes

            if not set(modes).issubset(available_modes):
                print(
                    "Requested {} unavailable modes".format(
                        " ".join(set(modes).difference(available_modes))
                    )
                )
                modes = list(set(modes).union(available_modes))
                print("Using modes {}".format(" ".join(modes)))

            h_lm = {(ell, m): h_lm[ell, m] for ell, m in modes}

        else:
            h_lm = self.h_lm
            times = self.times

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times
