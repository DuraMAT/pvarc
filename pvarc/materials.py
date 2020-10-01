
import numpy as np


def refractive_index_glass(wavelength, type='soda-lime-low-iron'):
    """
    Return real part of refractive index for glass given an array of
    wavelengths.

    Data for Soda-lime glass from:
     https://refractiveindex.info/?shelf=glass&book=soda-lime&page=Rubin-clear

    Rubin 1985. Range of validity is 310-4600 nm.

    Data for BK7 glass from:
        https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT

    Parameters
    ----------
    wavelength : ndarray
        wavelength in nm

    type : str
        Type of glass. Options are:

        'soda-lime-low-iron'

    Returns
    -------
    index_of_refraction : ndarray
        Index of refraction at specified wavelength.

    """

    if type.lower()== 'soda-lime-low-iron':
        wavelength = wavelength / 1000
        n = 1.5130 - 0.003169 * wavelength ** 2 + 0.003962 * wavelength ** -2
    elif type.upper()=='BK7':
        wavelength = wavelength / 1000
        n = np.sqrt(1 + \
                    (1.03961212 * wavelength ** 2) / (
                            wavelength ** 2 - 0.00600069867) + \
                    (0.231792344 * wavelength ** 2) / (
                            wavelength ** 2 - 0.0200179144) + \
                    (1.01046945 * wavelength ** 2) / (
                                wavelength ** 2 - 103.560653)
                    )


    return n


def refractive_index_porous_silica(wavelength, porosity=0.5):
    """
    Calculates index of refraction for porous silica using the effective
    medium approximation and volume averaging theory.

    https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson

    Parameters
    ----------
    wavelength : numeric

        Wavelength in nm.

    porosity : float

        Fractional porosity, a number from 0 to 1.0.


    Returns
    -------

    index : numeric

        Refractive index

    """
    wavelength = wavelength / 1000
    n = np.sqrt(1 + \
                (0.6961663 * wavelength ** 2) / (
                        wavelength ** 2 - 0.06840432 ** 2) + \
                (0.4079426 * wavelength ** 2) / (
                        wavelength ** 2 - 0.11624142 ** 2) + \
                (0.8974794 * wavelength ** 2) / (
                        wavelength ** 2 - 9.8961612 ** 2)
                )
    n_air = 1.00029

    n_total = np.sqrt(n ** 2 * (1 - porosity) + n_air ** 2 * (porosity))

    return n_total
