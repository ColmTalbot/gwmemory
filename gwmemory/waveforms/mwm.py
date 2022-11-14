from typing import Tuple

import numpy as np

from ..qnms import final_mass_spin, freq_damping
from ..utils import CC, GG, MPC, SOLAR_MASS
from . import MemoryGenerator


class MWM(MemoryGenerator):
    def __init__(
        self,
        q: float,
        total_mass: float = 60,
        distance: float = 400,
        name: str = "MWM",
        times: np.ndarray = None,
    ):
        super(MWM, self).__init__(name=name, h_lm=dict(), times=times)
        self.name = name
        if q > 1:
            q = 1 / q
        self.q = q
        self.MTot = total_mass
        self.distance = distance
        self.m1 = self.MTot / (1 + self.q)
        self.m2 = self.m1 * self.q

        if times is None:
            times = np.linspace(-900, 100, 10001) / self.t_to_geo
        self.times = times

    def time_domain_memory(
        self, inc: float, phase: float, times: np.ndarray = None, rm: float = 3
    ) -> Tuple[dict, np.ndarray]:
        """
        Calculates the plus and cross polarisations for the
        minimal waveform model memory waveform:
        eqns (5) and (9) from Favata (2009. ApJL 159)

        TODO: Implement spherical harmonic decomposition?

        Parameters
        ----------
        inc: float
            Binary inclination angle
        phase: float
            Binary phase at coalescence
        times: array, optional
            Time array on which the memory is calculated
        rm: float, optional
            Radius at which the matching occurs in solar masses

        Returns
        -------
        h_mem: dict
            Plus and cross polarisations of the memory waveform
        times: np.array
            Time array on which the memory is calculated

        Paul Lasky
        """
        if times is None:
            times = self.times

        time_geo = times * CC

        mass_solar_to_geometric = SOLAR_MASS * GG / CC**2
        m1_geo = self.m1 * mass_solar_to_geometric
        m2_geo = self.m2 * mass_solar_to_geometric

        dist_geo = self.distance * MPC

        # total mass
        MM = m1_geo + m2_geo

        # symmetric mass ratio
        eta = self.m1 * self.m2 / (self.m1 + self.m2) ** 2

        # this is the orbital separation at the matching radius --
        # see Favata (2009) before eqn (8).
        # the default value for this is given as rm = 3 MM.
        rm *= MM

        # calculate dimensionless mass and spin of the final black hole
        # from the Buonanno et al. (2007) fits
        Mf_geo, jj = final_mass_spin(m1_geo, m2_geo)

        # calculate the QNM frequencies and damping times
        # from the fits in Table VIII of Berti et al. (2006)
        omega, tau = np.array([freq_damping(Mf_geo, jj, nn=nn) for nn in range(3)]).T

        sigma = 1j * omega + 1 / tau
        sigma_plus_sigma_star = np.add.outer(sigma, sigma.conj())

        # set the matching time to be at t = 0
        tm = 0
        TT = time_geo - tm

        # some quantity defined after equation (7) of Favata
        trr = 5 * MM * rm**4 / (256 * eta * MM**4)

        # calculate the A_{ell m n} matching coefficients.  Note that
        # I've solved a matrix equation that solved for the three coefficients
        # from three equations
        xi = 2 * np.sqrt(2 * np.pi / 5) * eta * MM * rm**2
        chi = -2 * 1j * np.sqrt(MM / rm**3)

        amplitude = (
            xi
            * chi**2
            * np.array(
                [
                    (sigma[(ii + 1) % 3] + chi)
                    * (sigma[(ii + 2) % 3] + chi)
                    / (sigma[ii % 3] - sigma[(ii + 1) % 3])
                    / (sigma[ii % 3] - sigma[(ii + 2) % 3])
                    for ii in range(3)
                ]
            )
        )

        # Calculate the coefficients in the summed term of equation (9)
        # from Favata (2009) this is a double sum, with each variable going
        # from n = 0 to 2; therefore 9 terms
        coeffs = (
            np.outer(sigma * amplitude, sigma.conj() * amplitude.conj())
            / sigma_plus_sigma_star
        )

        # radial separation
        rr = rm * (1 - TT / trr) ** (1 / 4)

        # set up strain
        h_MWM = np.zeros(len(TT))

        # calculate strain for TT < 0.
        post_merger = TT > 0
        h_MWM[~post_merger] = 8 * np.pi * MM / rr[~post_merger]

        TT = TT[post_merger]

        # calculate strain for TT > 0.
        terms = coeffs * (
            1 - np.exp(-TT[:, np.newaxis, np.newaxis] * sigma_plus_sigma_star)
        )

        alt_sum_terms = terms.sum(axis=(1, 2))

        h_MWM[post_merger] = 8 * np.pi * MM / rm + alt_sum_terms / (eta * MM)

        # calculate the plus polarisation of GWs: eqn. (5) from Favata (2009)
        h_plus_coeff = (
            0.77
            * eta
            * MM
            / (384 * np.pi)
            * np.sin(inc) ** 2
            * (17 + np.cos(inc) ** 2)
            / dist_geo
        )
        h_mem = dict(plus=h_plus_coeff * h_MWM, cross=np.zeros_like(h_MWM))
        h_mem["plus"] -= h_mem["plus"][0]

        return h_mem, times
