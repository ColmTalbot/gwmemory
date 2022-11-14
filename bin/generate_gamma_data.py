"""
Generate the geometric factors that we cache.

Usage:

$ python generate_gamma_data.py $lmax $output_directory

For the pacakged data, `lmax=4`.
"""

import sys
from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from gwmemory import angles

ell_max = int(sys.argv[1])
output_directory = sys.argv[2]
existing = angles.load_gamma()
new = deepcopy(existing)

lms = list()
for ell in range(2, ell_max + 1)[::-1]:
    for m in range(-ell, ell + 1)[::-1]:
        lms.append((ell, m))

for ell1, m1 in tqdm(lms):
    label_1 = f"{ell1}{m1}"
    for ell2, m2 in lms:
        label_2 = f"{ell2}{m2}"
        gammalmlm = np.array(
            [angles.analytic_gamma((ell1, m1), (ell2, m2), ell) for ell in range(2, 21)]
        )
        delta_m = m1 - m2
        new[str(delta_m)][label_1 + label_2] = np.real(gammalmlm)
for key in new:
    new[key].to_csv(
        f"{output_directory}/gamma_coefficients_delta_m_{key}.dat",
        sep="\t",
        index=False,
    )
