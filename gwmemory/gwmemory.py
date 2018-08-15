from . import waveforms
from . import utils
import inspect


def time_domain_memory(model=None, h_lm=None, times=None, q=None, MTot=None, S1=None, S2=None, distance=None,
                       inc=None, phase=None, **kwargs):
    """
    Calculate the time domain memory waveform according to __reference__.

    Example usage:

    Using NR surrogate waveform __reference__ for an edge-on non-spinning, equal-mass, binary
    at a distance of 400 Mpc.

    h_mem, times = time_domain_memory(model='NRSur7dq2', q=1, MTot=60, distance=400, inc=np.pi/2, phase=0)


    Using an EOBNR waveform __reference__ for an edge-on non-spinning, equal-mass, binary
    at a distance of 400 Mpc.

    h_mem, times = time_domain_memory(model='SEOBNRv4', q=1, MTot=60, distance=400, inc=np.pi/2, phase=0)


    Using the minimal waveform model __reference__ for an edge-on non-spinning, equal-mass, binary
    at a distance of 400 Mpc.

    h_mem, times = time_domain_memory(model='MWM', q=1, MTot=60, distance=400, inc=np.pi/2, phase=0)


    Using a pre-computed spherical harmonic decomposed waveform for an edge-on non-spinning, equal-mass,
    binary at a distance of 400 Mpc.

    h_mem, times = time_domain_memory(h_lm=h_lm, times=times, distance=400, inc=np.pi/2, phase=0)


    Parameters
    ----------
    model: str
        Name of the model, this is used to identify waveform approximant, e.g., NRSur7dq2, IMRPhenomD, MWM, etc.
    h_lm: dict
        Spin weighted spherical harmonic decomposed time series.
        If this is specified these polarisations will be used.
    times: array
        time series corresponding to the h_lm.
    q: float
        Mass ratio of the binary being considered.
    MTot: float
        Total mass of the binary being considered in solar units.
    S1: array
        Dimensionless spin vector of the more massive black hole.
    S2: array
        Dimensionless spin vector of the less massive black hole.
    distance: float
        Distance to the binary in Mpc.
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
        Memory time series, either in spherical harmonic modes or plus/cross polarisations.
    times, array
        Time series corresponding to the memory waveform.
    """
    if h_lm is not None and times is not None:
        wave = waveforms.MemoryGenerator(name=model, h_lm=h_lm, times=times)
    elif 'NRSur' in model:
        model_kwargs =\
            {key: kwargs[key] for key in inspect.getargspec(waveforms.Surrogate.__init__)[0] if key in kwargs}
        wave = waveforms.Surrogate(q=q, name=model, MTot=MTot, S1=S1, S2=S2,
                                   distance=distance, times=times, **model_kwargs)
    elif 'EOBNR' in model or 'Phenom' in model:
        model_kwargs =\
            {key: kwargs[key] for key in inspect.getargspec(waveforms.Approximant.__init__)[0] if key in kwargs}
        wave = waveforms.Approximant(q=q, name=model, MTot=MTot, S1=S1, S2=S2,
                                     distance=distance, times=times, **model_kwargs)
    elif model == 'MWM':
        model_kwargs = {key: kwargs[key] for key in inspect.getargspec(waveforms.MWM.__init__)[0] if key in kwargs}
        wave = waveforms.MWM(q=q, name=model, MTot=MTot, distance=distance, times=times, **model_kwargs)
    else:
        print('Model {} unknown'.format(model))
        return None

    function_kwargs = {key: kwargs[key] for key in inspect.getargspec(wave.time_domain_memory)[0] if key in kwargs}
    h_mem, times = wave.time_domain_memory(inc=inc, phase=phase, **function_kwargs)

    return h_mem, times


def frequency_domain_memory(model=None, q=None, MTot=None, S1=None, S2=None, distance=None, inc=None, phase=None,
                            **kwargs):
    """
    Calculate the frequency domain memory waveform according to __reference__.

    Parameters
    ----------
    model: str
        Name of the model, this is used to identify waveform approximant, e.g., NRSur7dq2, IMRPhenomD, MWM, etc.
    h_lm: dict
        Spin weighted spherical harmonic decomposed time series.
        If this is specified these polarisations will be used.
    times: array
        time series corresponding to the h_lm.
    q: float
        Mass ratio of the binary being considered.
    MTot: float
        Total mass of the binary being considered in solar units.
    S1: array
        Dimensionless spin vector of the more massive black hole.
    S2: array
        Dimensionless spin vector of the less massive black hole.
    distance: float
        Distance to the binary in Mpc.
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
    h_mem, dict
        Memory frequency series, either in spherical harmonic modes or plus/cross polarisations.
    frequencies, array
        Frequency series corresponding to the memory waveform.
    """
    time_domain_strain, times = time_domain_memory(model=model, q=q, MTot=MTot, S1=S1, S2=S2, distance=distance,
                                                   inc=inc, phase=phase, **kwargs)
    sampling_frequency = 1 / (times[1] - times[0])

    frequency_domain_strain = dict()
    for key in time_domain_strain:
        frequency_domain_strain[key], frequencies = utils.nfft(time_domain_strain[key], sampling_frequency)

    return frequency_domain_strain, frequencies
