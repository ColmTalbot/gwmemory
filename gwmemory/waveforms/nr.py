from copy import deepcopy
from typing import Tuple

import numpy as np

from ..utils import combine_modes, load_sxs_waveform
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
        name: str,
        modes: list = None,
        extraction: str = "OutermostExtraction.dir",
        total_mass: float = None,
        distance: float = None,
        times: np.ndarray = None,
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
        distance: float
            Luminosity distance to the binary in MPC.
        times: array
            Time array to evaluate the waveforms on, default is time array
            in h5 file.
        """
        h_lm, times = load_sxs_waveform(name, modes=modes, extraction=extraction)
        super(SXSNumericalRelativity, self).__init__(
            name=name, h_lm=h_lm, times=times, l_max=4
        )

        self.MTot = total_mass
        self.distance = distance

        for mode in self.h_lm:
            self.h_lm[mode] /= self.h_to_geo
        self.times /= self.t_to_geo

        if times is not None:
            self.set_time_array(times)

    def time_domain_oscillatory(
        self,
        times: np.ndarray = None,
        modes: list = None,
        inc: float = None,
        phase: float = None,
    ) -> Tuple[dict, np.ndarray]:
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
        output = deepcopy(self.h_lm)
        if modes is not None:
            for mode in list(output.keys()):
                if mode not in modes:
                    del output[mode]
        if times is not None:
            output = self.apply_time_array(times, output)
        else:
            times = self.times
        if inc is None or phase is None:
            return output, times
        else:
            return combine_modes(output, inc, phase), times
