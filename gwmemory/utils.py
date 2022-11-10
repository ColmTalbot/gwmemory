import numpy as np

from .harmonics import sYlm, lmax_modes


# Constants for conversions
# taken from astropy==5.0.1
CC = 299792458.0
GG = 6.6743e-11
SOLAR_MASS = 1.988409870698051e+30
KG = 1 / SOLAR_MASS
METRE = CC ** 2 / (GG * SOLAR_MASS)
SECOND = CC * METRE

MPC = 3.085677581491367e+22


def nfft(ht, sampling_frequency):
    """
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    Parameters
    ----------
    ht: array-like
        Time series to FFT
    sampling_frequency: float
        Sampling frequency of input time series

    Returns
    -------
    hf: array-like
        Single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f: array-like
        Frequencies associated with hf
    """
    # add one zero padding if time series does not have even number
    # of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = sampling_frequency / 2 * np.linspace(0, 1, int(LL / 2 + 1))

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / sampling_frequency

    return hf, ff


def load_sxs_waveform(file_name, modes=None, extraction="OutermostExtraction.dir"):
    """
    Load the spherical harmonic modes of an SXS numerical relativity waveform.

    Parameters
    ----------
    file_name: str
        Name of file to be loaded.
    modes: dict
        Dictionary of spherical harmonic modes to extract,
        default is all in ell<=4.
    extraction: str
        String representing extraction method, default is
        'OutermostExtraction.dir'

    Returns
    -------
    output: dict
        Dictionary of requested spherical harmonic modes.
    """
    import h5py

    output = dict()
    with h5py.File(file_name, "r") as ff:
        waveform = ff[extraction]
        if modes is None:
            modes = lmax_modes(4)
        for (ell, mm) in modes:
            mode_array = waveform[f"Y_l{ell}_m{mm}.dat"][()]
            output[(ell, mm)] = mode_array[:, 1] + 1j * mode_array[:, 2]
        times = mode_array[:, 0]
    return output, times


def combine_modes(h_lm, inc, phase):
    """
    Calculate the plus and cross polarisations of the waveform from the
    spherical harmonic decomposition.
    """
    total = sum([h_lm[(l, m)] * sYlm(-2, l, m, inc, phase) for l, m in h_lm])
    h_plus_cross = dict(plus=total.real, cross=-total.imag)
    return h_plus_cross
