import numpy as np
import pandas as pd
import os


def get_AM1p5_spectrum():
    """
    Get the AM1.5 spectrum. Data file available at:

    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

    Returns
    -------

    df : dataframe

        AM1.5 spectral data as a dataframe.

    """
    cd = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(cd, 'astmg173.csv'), skiprows=1)
    return df


def solar_weighted_photon_reflectance(wavelength,
                                     reflectance,
                                     wavelength_min=400,
                                     wavelength_max=1100,
                                    ):
    """
    Calculate the solar-weighted photon reflectance (SWPR). This is equal to
    the reflectance weighted by the photon flux. Note this function can be
    also be used to calculate solar-weighted photon transmittance.

    Parameters
    ----------
    wavelength : ndarray

        wavelength values in nm

    reflection : ndarray

        reflectance (or transmittance) at the wavelength values.

    wavelength_min : float

        Minimum wavelength limit to the integral.

    wavelength_max : float

        Maximum wavelength limit to the integral.

    Returns
    -------

    swpr : float

        solar weighted photon reflectance.

    """
    cd = os.path.dirname(os.path.abspath(__file__))

    sun = pd.read_csv(os.path.join(cd, 'astmg173.csv'), skiprows=1)

    cax = np.logical_and(wavelength > wavelength_min,
                         wavelength < wavelength_max)
    wavelength = wavelength[cax]
    reflectance = reflectance[cax]
    # wavelength = np.linspace(200,1100,20)

    AM1p5 = np.interp(wavelength, sun['wavelength nm'],
                      sun['Global tilt  W*m-2*nm-1'])


    dwavelength = np.diff(wavelength)
    dwavelength = np.append(dwavelength, dwavelength[-1])

    photon_energy = 1 / wavelength
    swpr = np.sum(dwavelength * AM1p5 / photon_energy * reflectance) / \
          np.sum(dwavelength * AM1p5 / photon_energy)
    return swpr

def solar_integral(wavelength, spectrum,y,wavelength_min=200, wavelength_max=1200):

    cax = np.logical_and(wavelength > wavelength_min,
                         wavelength < wavelength_max)

    wavelength = wavelength[cax]
    y = y[cax]
    spectrum = spectrum[cax]

    dwavelength = np.diff(wavelength)
    dwavelength = np.append(dwavelength, dwavelength[-1])

    photon_energy = 1 / wavelength
    y_solar_weighted = np.sum(dwavelength * spectrum  * y) / \
           np.sum(dwavelength * spectrum )

    return y_solar_weighted