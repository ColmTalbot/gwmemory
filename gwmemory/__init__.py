from . import angles, gwmemory, harmonics, qnms, utils, waveforms
from .gwmemory import time_domain_memory, frequency_domain_memory

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = "unknown"
