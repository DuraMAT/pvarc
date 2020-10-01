"""

This module can be used to compare the output of the tmm package to the
corresponding optimized functions.

"""

import numpy as np
import pandas as pd
from tmm import coh_tmm, unpolarized_RT
from time import time
from numpy import pi, inf, cos, sin, exp



def thick_slab_reflectance_tmm(polarization, index_substrate, aoi, wavelength):
    """
    Reflection from a thick slab of material.

    Parameters
    ----------
    polarization : str

        Light polarization, can be "s" or "p" or "mixed"

    index_substrate : (N,) ndarray

        Index of refraction of the slab of material evaluated at the
        wavelengths specified in the `wavelength` input.

    aoi : float

        Angle of incidence in degrees

    wavelength : (N,) ndarray

        wavelength in nm, must be the same length as index_substrate.

    Returns
    -------

    """
    wavelength = wavelength.astype('float')
    degree = pi / 180
    R = np.zeros_like(wavelength)
    if polarization in ['s', 'p']:
        for j in range(len(R)):
            R[j] = coh_tmm(polarization,
                           [1.0003, index_substrate[j]],
                           [inf, inf],
                           aoi * degree, wavelength[j])['R']
    elif polarization == 'mixed':
        for j in range(len(R)):
            R[j] = unpolarized_RT([1.0003, index_substrate[j]], [inf, inf],
                                  aoi * degree, wavelength[j])['R']
    else:
        raise Exception("polarization must be 's','p' or 'mixed'")
    return R


def thin_film_reflectance_tmm(polarization, index_film, index_substrate,
                             film_thickness, aoi,
                             wavelength,
                             vectorize=False):
    """
    Calculate reflection from a thin film on a substrate. dimensions in nm.


    Parameters
    ----------
    polarization
    index_film
    index_substrate
    film_thickness
    aoi
    wavelength
    vectorize

    Returns
    -------

    """
    degree = pi / 180
    wavelength = wavelength.astype('float')

    d_list = [np.inf, film_thickness, np.inf]

    index_air = 1.0003

    R = np.zeros_like(wavelength)
    if polarization in ['s', 'p']:
        for j in range(len(R)):
            n_list = [1.0003, index_film[j], index_substrate[j]]
            R[j] = coh_tmm(polarization,
                           n_list,
                           d_list, aoi * degree, wavelength[j])['R']
    elif polarization == 'mixed':

        if vectorize:

            def unpolarized_RT_func(index_film, index_substrate, aoi,
                                    film_thickness,
                                    wavelength, ):
                return unpolarized_RT(
                    n_list=[index_air, index_film, index_substrate],
                    d_list=[np.inf, film_thickness, np.inf],
                    th_0=aoi * degree,
                    lam_vac=wavelength)['R']

            unpolarized_RT_vectorized = np.vectorize(unpolarized_RT_func)
            R = unpolarized_RT_vectorized(index_film, index_substrate, aoi,
                                          film_thickness, wavelength)
            print('vectorized done')
        else:

            for j in range(len(R)):
                n_list = [index_air, index_film[j], index_substrate[j]]
                R[j] = unpolarized_RT(n_list,
                                      d_list,
                                      aoi * degree,
                                      wavelength[j])['R']

    else:
        raise Exception("polarization must be 's','p' or 'mixed'")
    return R