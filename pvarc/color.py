import numpy as np
import pandas as pd

import colour
from colour.colorimetry import (MultiSpectralDistributions, SpectralShape, sd_ones)

from colour import SpectralDistribution, CubicSplineInterpolator, Extrapolator, \
    XYZ_to_sRGB, sd_to_XYZ, XYZ_to_xyY, XYZ_to_CIECAM02, xyY_to_XYZ, SDS_ILLUMINANTS



from colour.utilities.array import is_uniform, interval

from pvarc.materials import refractive_index_porous_silica, refractive_index_glass
from pvarc import thin_film_reflectance


# from skimage.color import rgb2xyz, xyz2lab


from colour.colorimetry import (intermediate_lightness_function_CIE1976,
                                intermediate_luminance_function_CIE1976)
from colour.models import xy_to_xyY, xyY_to_XYZ, Jab_to_JCh, JCh_to_Jab
from colour.utilities import (from_range_1, from_range_100, to_domain_1,
                              to_domain_100, tsplit, tstack)


def XYZ_to_Lab(
        XYZ,
        XYZ_n,
        ):
    """
    Converts from *CIE XYZ* tristimulus values to *CIE L\\*a\\*b\\**
    colourspace.

    ** TODD KARIN: Modified to use XYZ_n rather than illuminant.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        *CIE L\\*a\\*b\\** colourspace array.

    Notes
    -----

    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``Lab``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |                |                       |                 |
    |                | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_Lab(XYZ)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  52.6385830...,  26.9231792...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))

    X_n, Y_n, Z_n = tsplit(XYZ_n)

    f_X_X_n = intermediate_lightness_function_CIE1976(X, X_n)
    f_Y_Y_n = intermediate_lightness_function_CIE1976(Y, Y_n)
    f_Z_Z_n = intermediate_lightness_function_CIE1976(Z, Z_n)

    L = 116 * f_Y_Y_n - 16
    a = 500 * (f_X_X_n - f_Y_Y_n)
    b = 200 * (f_Y_Y_n - f_Z_Z_n)

    Lab = tstack([L, a, b])

    return from_range_100(Lab)

def spectrum_to_XYZ(wavelength, spectrum,
                    # cmfs=None,
                    illuminant=None,
                    ):
    """
    Calculate the rgb color given a wavelength and spectrum

    Parameters
    ----------
    wavelength
    spectrum
    cmfs
    illuminant

    Returns
    -------

    """
    # if cmfs is None:
        # cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']



    if illuminant is None:
        # illuminant = ILLUMINANTS_SDS['CIE 1931 2 Degree Standard Observer']['D65']
        illuminant = sd_ones()
    elif type(illuminant)==type(''):
        illuminant = SDS_ILLUMINANTS[illuminant]
    # # Get illuminant
    # if type(illuminant) == type(''):
    #     illuminant = colour.ILLUMINANTS_SDS[illuminant]

    # Build spectral distribution object
    sd = SpectralDistribution(pd.Series(spectrum, index=wavelength),
                              interpolator=CubicSplineInterpolator,
                              extrapolator=Extrapolator)

    # Calculate xyz color coordinates.
    xyz = sd_to_XYZ(sd=sd,
                    illuminant=illuminant)

    return xyz


def spectrum_to_rgb(wavelength, spectrum,
                    # cmfs=None,
                    illuminant='LED-B3',
                    Y=None,
                    ):
    """
    Calculate the rgb color given a wavelength and spectrum. Note spectrum should have spacing of 1, 5, 10 or 10 nm.

    Parameters
    ----------
    wavelength
    spectrum
    cmfs
    illuminant

    Returns
    -------

    """

    # if is_uniform(wavelength) and interval(wavelength)
    XYZ = spectrum_to_XYZ(wavelength=wavelength,
                          spectrum=spectrum,
                          # cmfs=cmfs,
                          illuminant=illuminant)

    XYZ = XYZ / 100

    if Y is not None:
        xyY = XYZ_to_xyY(XYZ)
        xyY[2] = Y
        XYZ = xyY_to_XYZ(xyY)

    # Convert to rgb. Note the illuminant doesn't matter here except for points
    # where XYZ=0.
    rgb = XYZ_to_sRGB(XYZ)

    return rgb


def spectrum_to_xyY(wavelength, spectrum,
                    cmfs=None,
                    illuminant='LED-B3',
                    ):
    """
    Calculate the rgb color given a wavelength and spectrum

    Parameters
    ----------
    wavelength
    spectrum
    cmfs
    illuminant

    Returns
    -------

    """
    xyz = spectrum_to_XYZ(wavelength=wavelength,
                          spectrum=spectrum,
                          # cmfs=cmfs,
                          illuminant=illuminant)

    # Convert to rgb.
    # TODO: check on the divide by 100
    xyY = XYZ_to_xyY(xyz / 100)

    return xyY


# def spectrum_to_Lab(wavelength, spectrum,
#                     cmfs=None,
#                     illuminant='LED-B3',
#                     ):
#     xyz = spectrum_to_XYZ(wavelength=wavelength,
#                           spectrum=spectrum,
#                           cmfs=cmfs,
#                           illuminant=illuminant)
#
#     # Convert to rgb.
#     # TODO: check on the divide by 100
#     Lab = XYZ_to_Lab(xyz / 100)
#
#     return Lab


def arc_model_to_rgb(thickness, porosity, aoi=8):
    # Wavelength axis
    wavelength = np.arange(360, 801, 5).astype('float')

    # Choice of illuminant makes a very small change to the calculated RGB color.
    illuminant = 'LED-B3'

    # # Scan thickness and porosity.
    # thickness = np.arange(0, 196, 5).astype('float')
    # porosity = np.arange(0, 0.51, 0.1).astype('float')

    # col = np.zeros((3, len(thickness), len(porosity)))
    # col_hex = np.empty((len(thickness), len(porosity)), dtype='object')
    # xyY = np.zeros((3, len(thickness), len(porosity)))

    index_film = refractive_index_porous_silica(wavelength, porosity)
    index_substrate = refractive_index_glass(wavelength)

    # Calculate reflectance
    reflectance = thin_film_reflectance(index_film=index_film,
                                        index_substrate=index_substrate,
                                        film_thickness=thickness,
                                        aoi=aoi,
                                        wavelength=wavelength)
    reflectance_ref = thin_film_reflectance(index_film=index_film,
                                            index_substrate=index_substrate,
                                            film_thickness=0,
                                            aoi=aoi,
                                            wavelength=wavelength)

    rgb = spectrum_to_rgb(wavelength, reflectance,
                          illuminant=illuminant)
    rgb_ref = spectrum_to_rgb(wavelength, reflectance_ref,
                              illuminant=illuminant)

    # White balance
    rgb_wb = rgb / rgb_ref

    # Clamp
    rgb_wb[rgb_wb >= 1] = 1
    rgb_wb[rgb_wb < 0] = 0

    return rgb_wb

#
# def rgb2xyY(rgb):
#     xyz = rgb2xyz(rgb)
