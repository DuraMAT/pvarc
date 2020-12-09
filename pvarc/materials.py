import numpy as np
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def refractive_index_imaginary_silica(wavelength):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, 'materials', 'refractive_index_k_silica_Kitamura2007.csv')
    df = pd.read_csv(fname)
    print(df.keys())
    k = np.interp(wavelength, df['wavelength'] * 1000., df['k'])

    return k

def reflectance_PTFE(wavelength):


    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, 'materials', 'PMR10_PTFE_reflectance.xlsx')
    df = pd.read_excel(fname)
    reflectance =  np.interp(wavelength, df['Wavelength (nm)'], df['Reflectance (%)'])

    return reflectance


def refractive_index_fused_silica(wavelength):
    """
    Refractive index for Corning UV fused silica 7980

    https://www.corning.com/media/worldwide/csm/documents/5bf092438c5546dfa9b08e423348317b.pdf

    Parameters
    ----------
    wavelength
        wavelength in nm

    Returns
    -------

    """
    wavelength_um = wavelength / 1000

    A0 = 2.104025406E+00
    A1 = -1.456000330E-04
    A2 = -9.049135390E-03
    A3 = 8.801830992E-03
    A4 = 8.435237228E-05
    A5 = 1.681656789E-06
    A6 = -1.675425449E-08
    A7 = 8.326602461E-10

    n = np.sqrt( A0 + A1 * wavelength_um ** 4 + A2 * wavelength_um ** 2 + A3 * wavelength_um ** -2 + \
         A4 * wavelength_um ** -4 + A5 * wavelength_um ** -6 + A6 * wavelength_um ** -8 + A7 * wavelength_um ** -10 )

    return n


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

    if type.lower() == 'soda-lime-low-iron':
        wavelength = wavelength / 1000
        n = 1.5130 - 0.003169 * wavelength ** 2 + 0.003962 * wavelength ** -2 + 0 * 1j

        # n[wavelength < 0.3] = n[wavelength < 0.3] + 1j*0
    elif type.upper() == 'BK7':
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
    wavelength_um = wavelength / 1000
    n = np.sqrt(1 + \
                (0.6961663 * wavelength_um ** 2) / (
                        wavelength_um ** 2 - 0.06840432 ** 2) + \
                (0.4079426 * wavelength_um ** 2) / (
                        wavelength_um ** 2 - 0.11624142 ** 2) + \
                (0.8974794 * wavelength_um ** 2) / (
                        wavelength_um ** 2 - 9.8961612 ** 2)
                )
    n_air = 1.00029

    n_total = np.sqrt(n ** 2 * (1 - porosity) + n_air ** 2 * (porosity)) + 0 * 1j

    # k0 = 5e-6
    # k1 = 5e-7
    # wavelength0 = 0.31
    # wavelength1 = 0.36

    # n_total = n_total + 1j*refractive_index_imaginary_silica(wavelength)*1e4
    # n_total = n_total + 1j*np.exp( np.log(k0) + np.log(k1) * (wavelength - wavelength0)/(wavelength1-wavelength0))

    return n_total
