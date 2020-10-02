import pandas as pd

import colour
from colour import SpectralDistribution, CubicSplineInterpolator, Extrapolator, \
    XYZ_to_sRGB, sd_to_XYZ

def spectrum_to_rgb(wavelength, spectrum,
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
    if cmfs is None:
        cmfs = colour.STANDARD_OBSERVERS_CMFS[
            'CIE 1931 2 Degree Standard Observer']

    # Get illuminant
    if type(illuminant) == type(''):
        illuminant = colour.ILLUMINANTS_SDS[illuminant]

    # Build spectral distribution object
    sd = SpectralDistribution(pd.Series(spectrum, index=wavelength),
                              interpolator=CubicSplineInterpolator,
                              extrapolator=Extrapolator)

    # Calculate xyz color coordinates.
    xyz = sd_to_XYZ(sd, cmfs, illuminant)

    # Convert to rgb.
    rgb = XYZ_to_sRGB(xyz / 100)

    return rgb