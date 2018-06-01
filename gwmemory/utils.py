import numpy as np
import deepdish


# Constants for conversions
cc = 299792458.0  # speed of light in m/s
GG = 6.67384e-11  # Newton in m^3 / (kg s^2)
solar_mass = 1.98855 * 10**(30)  # solar mass in  kg
kg = 1 / solar_mass
metre = cc**2 / (GG * solar_mass)
secs = cc * metre

Mpc = 3.08568e+22  # Mpc in metres


def m12_to_mc(m1, m2):
    """convert m1 and m2 to chirp mass"""
    return (m1*m2)**(3./5.) / (m1 + m2)**(1./5.)


def m12_to_symratio(m1, m2):
    # convert m1 and m2 to symmetric mass ratio
    return m1 * m2 / (m1 + m2)**2


def mc_eta_to_m12(mc, eta):
    """
    Convert chirp mass and symmetric mass ratio to component masses.

    Input: mc - chirp mass
    eta - symmetric mass ratio
    Return: m1, m2 - primary and secondary masses, m1>m2
    """
    m1 = mc/eta**0.6*(1+(1-4*eta)**0.5)/2
    m2 = mc/eta**0.6*(1-(1-4*eta)**0.5)/2
    return m1, m2


def m_sol_to_geo(mm):
    """convert from solar masses to geometric units"""
    return mm / kg * GG / cc**2


def m_geo_to_sol(mm):
    """convert from geometric units to solar masses"""
    return mm * kg / GG * cc**2


def time_s_to_geo(time):
    """convert time from seconds to geometric units"""
    return time * cc


def time_geo_to_s(time):
    """convert time from seconds to geometric units"""
    return time / cc


def freq_Hz_to_geo(freq):
    """convert freq from Hz to geometric units"""
    return freq / cc


def freq_geo_to_Hz(freq):
    """convert freq from geometric units to Hz"""
    return freq * cc


def dist_Mpc_to_geo(dist):
    """convert distance from Mpc to geometric units (i.e., metres)"""
    return dist * Mpc


def nfft(ht, sampling_frequency):
    """
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    ht = time series
    sampling_frequency = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    """
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = sampling_frequency / 2 * np.linspace(0, 1, int(LL/2+1))

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / sampling_frequency

    return hf, ff


def load_sxs_waveform(file_name, modes=None, extraction='OutermostExtraction.dir'):
    """
    Load the spherical harmonic modes of an SXS numerical relativity waveform.

    Parameters
    ----------
    file_name: str
        Name of file to be loaded.
    modes: dict
        Dictionary of spherical harmonic modes to extract, default is all in ell<=4.
    extraction: str
        String representing extraction method, default is 'OutermostExtraction.dir'
    Returns
    -------
    output: dict
        Dictionary of requested spherical harmonic modes.
    """
    waveform = deepdish.io.load(file_name)
    if modes is None:
        output = dict()
        for ell in range(2, 5):
            for mm in range(-ell, ell + 1):
                mode_array = waveform[extraction]['Y_l{}_m{}.dat'.format(ell, mm)[:, 1:]]
                output[(ell, mm)] = mode_array[:, 1] + 1j * mode_array[:, 2]
    else:
        for mode in modes:
            mode_array = waveform[extraction]['Y_l{}_m{}.dat'.format(mode[0], mode[1])]
            output[mode] = mode_array[:, 1] + 1j * mode_array[:, 2]
    times = mode_array[:, 0]
    return output, times
