import inspect

from .base import MemoryGenerator  # isort: skip
from . import approximant, mwm, nr, surrogate  # isort: skip
from .approximant import Approximant  # isort: skip
from .mwm import MWM  # isort: skip
from .nr import SXSNumericalRelativity  # isort: skip
from .surrogate import Surrogate  # isort: skip


GENERATORS = dict(
    base=MemoryGenerator,
    EOBNR=Approximant,
    MWM=MWM,
    NRHybSur=Surrogate,
    NRSur=Surrogate,
    Phenom=Approximant,
    SXS=SXSNumericalRelativity,
)


def memory_generator(model, **kwargs):
    """
    Create a memory generator from any of the registered classes.

    Parameters
    ==========
    model: str
        The name of the model to use.
    kwargs:
        Arguments to pass to the :code:`__init__` method of the generator.

    Returns
    =======
    MemoryGenerator
        The memory generator instance.
    """
    cls_ = None
    for key, value in GENERATORS.items():
        if key in model:
            cls_ = value
    if cls_ is None:
        raise ValueError(
            f"Unknown waveform generator {model}. "
            "Should match one of {GENERATORS.keys()}"
        )
    all_keys = inspect.signature(cls_).parameters.keys()
    model_kwargs = {key: kwargs[key] for key in all_keys if key in kwargs}
    return cls_(**model_kwargs)


def register_generator(model: str, generator: MemoryGenerator):
    """
    Register a new memory generator.
    If you have implemented a new memory generator (that matches the API)
    here, you can use this to automatically have it be found by
    :code:`gwmemory.time_domain_memory`.

    Parameters
    ==========
    model: str
        The name to register the model as.
    generator: MemoryGenerator
        The new model to register.
    """
    import warnings

    if model in GENERATORS:
        warnings.warn(f"Overwriting previously registered model {model}.")
    else:
        for key in GENERATORS:
            if key in model:
                warnings.warn(f"Collision with existing model {key}.")
    GENERATORS[model] = generator
