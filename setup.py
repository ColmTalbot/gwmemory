#!/usr/bin/env python

from distutils.core import setup

setup(name='gwmemory',
      version='0.1.0',
      packages=['gwmemory'],
      package_dir={'gwmemory': 'gwmemory'},
      package_data={'gwmemory': ['data/gamma_coefficients*.dat', 'data/*WEB.dat']},
      )
