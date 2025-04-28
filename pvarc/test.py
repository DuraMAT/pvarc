"""

This module can be used to compare the output of the tmm package to the
corresponding optimized functions.

"""

import numpy as np
import pandas as pd
from tmm import coh_tmm, unpolarized_RT
from time import time
from numpy import pi, inf, cos, sin, exp

from pvarc.main import double_film_reflectance


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



def compare_tmm_vs_double_film(double_film_reflectance_func):
    """
    Compare 'double_film_reflectance' results with the 'tmm' library
    for a simple 2-film stack: air | film1 | film2 | substrate.

    Parameters
    ----------
    double_film_reflectance_func : callable
        Your function with signature:
           R, T, A = double_film_reflectance_func(index_film1, index_film2, index_substrate,
                                                 thickness_film1, thickness_film2,
                                                 wavelength, aoi=..., polarization=...,
                                                 index_air=...)
        that returns arrays of R, T, A.

    Returns
    -------
    None
        Prints out or asserts that the two methods agree closely.
    """
    from tmm import coh_tmm, unpolarized_RT

    # -----------------------------
    # 1) Define a two-film stack
    # -----------------------------
    n0 = 1.0          # ambient
    n1 = 1.45         # film1 index
    d1_nm = 100.0     # film1 thickness in nm
    n2 = 2.00         # film2 index
    d2_nm = 50.0      # film2 thickness in nm
    n3 = 3.7          # substrate (semi-infinite)

    aoi_deg = 8.0     # angle of incidence in degrees
    polarization = 's'  # or 'p' or 'mixed'

    # Wavelength array in nm
    wls_nm = np.linspace(400, 800, 21)  # 21 points from 400 nm to 800 nm

    # -----------------------------
    # 2) Use double_film_reflectance
    #    We expect it can handle vector (array) input for wavelength
    # -----------------------------
    R_df, T_df, A_df = double_film_reflectance_func(
        index_film1=n1,
        index_film2=n2,
        index_substrate=n3,
        thickness_film1=d1_nm,
        thickness_film2=d2_nm,
        wavelength=wls_nm,
        aoi=aoi_deg,
        polarization=polarization,
        index_air=n0
    )

    # Make sure we have arrays
    R_df = np.array(R_df)
    T_df = np.array(T_df)
    A_df = np.array(A_df)

    # -----------------------------
    # 3) Use tmm to get R, T, A
    # -----------------------------
    # tmm wants thickness in microns, and lam_vac in microns
    # Also needs an array of n_list, d_list where the top and bottom are 'semi-infinite'
    # We'll do: d_list = [inf, film1_thick_um, film2_thick_um, inf]
    d1_um = d1_nm * 1e-3
    d2_um = d2_nm * 1e-3

    # For tmm, define layer thickness array (semi-infinite for first and last):
    # layer 0 (top) = air => thickness = np.inf
    # layer 1 = film1 => thickness = d1_um
    # layer 2 = film2 => thickness = d2_um
    # layer 3 (bottom) = substrate => thickness = np.inf
    d_tmm = [np.inf, d1_um, d2_um, np.inf]

    # The n_list for tmm
    # must match the layers: [air, film1, film2, substrate]
    n_tmm = [n0, n1, n2, n3]

    # We'll store results in arrays for tmm
    R_tmm = np.zeros_like(wls_nm)
    T_tmm = np.zeros_like(wls_nm)
    A_tmm = np.zeros_like(wls_nm)

    # Convert angle to radians
    aoi_rad = np.deg2rad(aoi_deg)

    # We loop over each wavelength to call tmm.coh_tmm
    for i, w_nm in enumerate(wls_nm):
        # tmm expects vacuum wavelength in microns:
        lam_vac_um = w_nm * 1e-3

        # call tmm
        res = coh_tmm(
            pol=polarization,    # 's' or 'p'
            n_list=n_tmm,
            d_list=d_tmm,
            th_0=aoi_rad,
            lam_vac=lam_vac_um
        )
        R_tmm[i] = res['R']
        T_tmm[i] = res['T']
        A_tmm[i] = 1.0 - R_tmm[i] - T_tmm[i]

    # -----------------------------
    # 4) Print or compare
    # -----------------------------
    print("\nComparison of double_film_reflectance vs tmm package:")
    for i, w_nm in enumerate(wls_nm):
        print(f"λ={w_nm:5.1f} nm:  DF => R={R_df[i]:.4f}, T={T_df[i]:.4f}, A={A_df[i]:.4f} "
              f" |  TMM => R={R_tmm[i]:.4f}, T={T_tmm[i]:.4f}, A={A_tmm[i]:.4f}")

    # Optionally, we can check closeness:
    # e.g. using np.allclose or np.max(abs(diff))
    diff_R = np.max(np.abs(R_df - R_tmm))
    diff_T = np.max(np.abs(T_df - T_tmm))
    diff_A = np.max(np.abs(A_df - A_tmm))
    print(f"\nMax difference in R: {diff_R:.3e}")
    print(f"Max difference in T: {diff_T:.3e}")
    print(f"Max difference in A: {diff_A:.3e}")

    # If you'd like to enforce a tolerance:
    # e.g. assert diff_R < 1e-5, etc.


# --------------------------------------------
# Example usage, if double_film_reflectance is already defined
# --------------------------------------------
if __name__ == "__main__":


    compare_tmm_vs_double_film(double_film_reflectance)
