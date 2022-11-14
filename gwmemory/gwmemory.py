import inspect
from typing import Tuple

import numpy as np

from . import utils, waveforms


def time_domain_memory(
    model: str = None,
    h_lm: dict = None,
    times: np.ndarray = None,
    q: float = None,
    total_mass: float = None,
    spin_1: Tuple[float, float, float] = None,
    spin_2: Tuple[float, float, float] = None,
    distance: float = None,
    inc: float = None,
    phase: float = None,
    **kwargs,
) -> Tuple[dict, np.ndarray]:
    """
    Calculate the time domain memory waveform according to __reference__.

    Example usage:

    Using NR surrogate waveform __reference__ for an edge-on non-spinning,
    equal-mass, binary at a distance of 400 MPC.

    h_mem, times = time_domain_memory(model='NRSur7dq2', q=1, total_mass=60,
                                      distance=400, inc=np.pi/2, phase=0)


    Using an EOBNR waveform __reference__ for an edge-on non-spinning,
    equal-mass, binary at a distance of 400 MPC.

    h_mem, times = time_domain_memory(model='SEOBNRv4', q=1, total_mass=60,
                                      distance=400, inc=np.pi/2, phase=0)


    Using the minimal waveform model __reference__ for an edge-on non-spinning,
    equal-mass, binary at a distance of 400 MPC.

    h_mem, times = time_domain_memory(model='MWM', q=1, total_mass=60,
                                      distance=400, inc=np.pi/2, phase=0)


    Using a pre-computed spherical harmonic decomposed waveform for an edge-on
    non-spinning, equal-mass, binary at a distance of 400 MPC.

    h_mem, times = time_domain_memory(h_lm=h_lm, times=times, distance=400,
                                      inc=np.pi/2, phase=0)


    Parameters
    ----------
    model: str
        Name of the model, this is used to identify waveform approximant,
        e.g., NRSur7dq2, IMRPhenomD, MWM, etc.
    h_lm: dict
        Spin weighted spherical harmonic decomposed time series.
        If this is specified these polarisations will be used.
    times: array
        time series corresponding to the h_lm.
    q: float
        Mass ratio of the binary being considered.
    total_mass: float
        Total mass of the binary being considered in solar units.
    spin_1: array
        Dimensionless spin vector of the more massive black hole.
    spin_2: array
        Dimensionless spin vector of the less massive black hole.
    distance: float
        Distance to the binary in MPC.
    inc: float
        Inclination of the binary to the line of sight.
        If not provided, spherical harmonic modes will be returned.
    phase: float
        Binary phase as coalescence.
        If not provided, spherical harmonic modes will be returned.
    kwargs: dict
        Additional model-specific keyword arguments.

    Returns
    -------
    h_mem, dict
        Memory time series, either in spherical harmonic modes or plus/cross
        polarisations.
    times, array
        Time series corresponding to the memory waveform.
    """
    if h_lm is not None and times is not None:
        wave = waveforms.MemoryGenerator(name=model, h_lm=h_lm, times=times)
    elif "NRSur" in model or "NRHybSur" in model:
        all_keys = inspect.signature(waveforms.Surrogate).parameters.keys()
        model_kwargs = {key: kwargs[key] for key in all_keys if key in kwargs}
        wave = waveforms.Surrogate(
            q=q,
            name=model,
            total_mass=total_mass,
            spin_1=spin_1,
            spin_2=spin_2,
            distance=distance,
            times=times,
            **model_kwargs,
        )
    elif "EOBNR" in model or "Phenom" in model:
        all_keys = inspect.signature(waveforms.Approximant).parameters.keys()
        model_kwargs = {key: kwargs[key] for key in all_keys if key in kwargs}
        wave = waveforms.Approximant(
            q=q,
            name=model,
            total_mass=total_mass,
            spin_1=spin_1,
            spin_2=spin_2,
            distance=distance,
            times=times,
            **model_kwargs,
        )
    elif model == "MWM":
        all_keys = inspect.signature(waveforms.MWM).parameters.keys()
        model_kwargs = {key: kwargs[key] for key in all_keys if key in kwargs}
        wave = waveforms.MWM(
            q=q,
            name=model,
            total_mass=total_mass,
            distance=distance,
            times=times,
            **model_kwargs,
        )
    else:
        print(f"Model {model} unknown")
        return None

    all_keys = inspect.signature(wave.time_domain_memory).parameters.keys()
    function_kwargs = {key: kwargs[key] for key in all_keys if key in kwargs}
    h_mem, times = wave.time_domain_memory(inc=inc, phase=phase, **function_kwargs)

    return h_mem, times


def frequency_domain_memory(
    model: str = None,
    q: float = None,
    total_mass: float = None,
    spin_1: Tuple[float, float, float] = None,
    spin_2: Tuple[float, float, float] = None,
    distance: float = None,
    inc: float = None,
    phase: float = None,
    **kwargs,
) -> Tuple[dict, np.ndarray]:
    """
    Calculate the frequency domain memory waveform according to __reference__.

    Parameters
    ----------
    model: str
        Name of the model, this is used to identify waveform approximant,
        e.g., NRSur7dq2, IMRPhenomD, MWM, etc.
    q: float
        Mass ratio of the binary being considered.
    total_mass: float
        Total mass of the binary being considered in solar units.
    spin_1: array
        Dimensionless spin vector of the more massive black hole.
    spin_2: array
        Dimensionless spin vector of the less massive black hole.
    distance: float
        Distance to the binary in MPC.
    inc: float
        Inclination of the binary to the line of sight.
        If not provided, spherical harmonic modes will be returned.
    phase: float
        Binary phase at coalescence.
        If not provided, spherical harmonic modes will be returned.
    kwargs: dict
        Additional model-specific keyword arguments.

    Returns
    -------
    frequency_domain_strain: dict
        Memory frequency series, either in spherical harmonic modes or
        plus/cross polarisations.
    frequencies: array-like
        Frequency series corresponding to the memory waveform.
    """
    time_domain_strain, times = time_domain_memory(
        model=model,
        q=q,
        total_mass=total_mass,
        spin_1=spin_1,
        spin_2=spin_2,
        distance=distance,
        inc=inc,
        phase=phase,
        **kwargs,
    )
    sampling_frequency = 1 / (times[1] - times[0])

    frequencies = None
    frequency_domain_strain = dict()
    for key in time_domain_strain:
        frequency_domain_strain[key], frequencies = utils.nfft(
            time_domain_strain[key], sampling_frequency
        )

    return frequency_domain_strain, frequencies
