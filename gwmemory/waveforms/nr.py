import numpy as np

from ..utils import combine_modes, load_sxs_waveform, CC, GG, MPC, SOLAR_MASS
from . import MemoryGenerator


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

    def __init__(
        self,
        name,
        modes=None,
        extraction="OutermostExtraction.dir",
        total_mass=None,
        distance=None,
        times=None,
    ):
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
        self.h_lm, self.times = load_sxs_waveform(
            name, modes=modes, extraction=extraction
        )

        self.MTot = total_mass
        self.distance = distance

        if total_mass is None or distance is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * MPC / self.MTot / SOLAR_MASS / GG * CC ** 2
            self.t_to_geo = 1 / self.MTot / SOLAR_MASS / GG * CC ** 3

            for mode in self.h_lm:
                self.h_lm /= self.h_to_geo
            self.times / self.t_to_geo
            # Rezero time array to the merger time
            self.times -= self.times[np.argmax(abs(self.h_lm[(2, 2)]))]

        if times is not None:
            self.set_time_array(times)

        MemoryGenerator.__init__(self, name=name, h_lm=self.h_lm, times=times)

    def time_domain_oscillatory(self, times=None, modes=None, inc=None, phase=None):
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
