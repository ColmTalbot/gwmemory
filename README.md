# `GWMemory`

`GWMemory` calculates the nonlinear (Christodoulou) gravitational-wave memory waveform from arbitrary time-domain gravitational-waveforms.

## Installation

You can install from this repository in the usual way:

```bash
$ git clone https://github.com/ColmTalbot/gwmemory.git
$ cd gwmemory
$ python -m pip install .
```

Note that it is no longer to install using
```bash
$ python setup.py install
```
The `pip` version is more stable.

**TODO: make pip/conda installable**

## Examples

Demonstrations of how to calculate memory waveforms can be found in the examples directory.

## Supported waveforms

- [NRSur7dq2](https://zenodo.org/record/1215824#.WzcMDuiFPEY) (Blackman _et al._ (2017), [Phys. Rev. D 96, 024058](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.024058))
- Aligned spin waveforms in [`lalsimulation`](http://git.ligo.org/lscsoft/lalsuite), e.g., `IMRPhenomD` ( Khan _et al._ (2016), [Phys. Rev. D 93, 044007](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.044007), `SEOBNRv4` (Bohe _et al._ (2017), [Phys. Rev. D 95, 044028](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.044028)).
- Minimal waveform model (Favata (2010), [CQG, 27, 8](http://iopscience.iop.org/article/10.1088/0264-9381/27/8/084036/meta))
- Any surrogate model available in `gwsurrogate`

To use these waveform models, you may need to install extra packages, this should be described if the packages aren't available.

Additionally users can supply any waveform decomposed onto the basis of spin-2 weighted spherical harmonics.

There is basic support for loading numerical relativity waveforms, based on the format of the simulating extreme spacetimes [waveform catalog](https://www.black-holes.org/for-researchers/waveform-catalog).
