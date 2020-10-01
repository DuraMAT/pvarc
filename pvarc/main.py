import numpy as np
from numpy import pi, cos, sin


def thin_film_reflectance(index_film,
                          index_substrate,
                          film_thickness,
                          wavelength,
                          aoi=8.0,
                          polarization='mixed',
                          index_air=1.0003):
    """
    Calculate reflectance spectrum from a thin film. The optical stack
    consists of a semi-infinite layer of air, a thin film with a given
    thickness, and a semi-infinite substrate layer.

    Inputs can be float or ndarray. All inputs that are ndarray must have the
    same shape. For example, if an array of wavelength values with shape (1,
    N) is passed, then index_film can either be a float or an ndarray with
    shape (1,N).

    The method is derived from the matrix form of the Fresnel equations,
    detailed in Ref. [1].

    [1] https://arxiv.org/abs/1603.02720

    Parameters
    ----------

    index_film : ndarray or float

        Refractive index of the thin film at the values specified in the
        `wavelength` input.

    index_substrate : ndarray or float

        Refractive index of the substrate at the values specified in the
        `wavelength` input.

    film_thickness : ndarray or float

        Thickness of thin film in nm.

    wavelength : ndarray or float

        Wavelength in nm

    aoi : ndarray or float

        Angle of incidence of light onto thin film, in degrees.

    polarization : str

        Polarization of light, can be 's', 'p' or 'mixed'.

    index_air : ndarray or float

        Index of refraction of air, default is 1.0003

    Returns
    -------

    """

    # Calculate angles from Snell's law
    theta0 = aoi * pi / 180
    theta1 = np.arcsin(index_air / index_film * sin(theta0))
    theta2 = np.arcsin(index_air / index_substrate * sin(theta0))

    # Give index of refraction values better names.
    n0 = index_air
    n1 = index_film
    n2 = index_substrate

    # Calculate reflectance amplitudes
    rs01 = (n0 * cos(theta0) - n1 * cos(theta1)) / (
            n0 * cos(theta0) + n1 * cos(theta1))

    rp01 = (n1 * cos(theta0) - n0 * cos(theta1)) / (
            n1 * cos(theta0) + n0 * cos(theta1))

    rs12 = (n1 * cos(theta1) - n2 * cos(theta2)) / (
            n1 * cos(theta1) + n2 * cos(theta2))

    rp12 = (n2 * cos(theta1) - n1 * cos(theta2)) / (
            n2 * cos(theta1) + n1 * cos(theta2))

    # delta is used in film thickness.
    kz = 2 * pi * n1 * cos(theta1) / wavelength
    delta = film_thickness * kz

    # Calculate reflected power for s and p light.
    Rs = np.abs(
        np.exp(-1j * delta) * rs12 + np.exp(1j * delta) * rs01) ** 2 / np.abs(
        np.exp(-1j * delta) + np.exp(1j * delta) * rs01 * rs12) ** 2

    Rp = np.abs(
        np.exp(-1j * delta) * rp12 + np.exp(1j * delta) * rp01) ** 2 / np.abs(
        np.exp(-1j * delta) + np.exp(1j * delta) * rp01 * rp12) ** 2

    # Return reflectance depending on polarization chosen.
    if polarization == 'mixed':
        return 0.5 * (Rs + Rp)
    elif polarization == 's':
        return Rs
    elif polarization == 'p':
        return Rp
    else:
        raise Exception('Polarization must be "mixed", "s" or "p".')




def single_interface_reflectance(n0,n1, aoi=8.0, polarization='mixed'):
    """
    Reflection from a single interface for a wave traveling from material n0
    to n1. Both n0 and n1 can be an array of values.

    Parameters
    ----------
    n0 : ndarray or float
        Index of refraction for semi-infinite material with incident wave.

    n1 : ndarray or float
        Index of refraction for semi-infinite material that wave reflects from.

    aoi : ndarray or float
        Angle of incidence in degrees.

    polarization : str
        Polarization of wave, can be 's', 'p' or 'mixed'.

    Returns
    -------
    reflectance : ndarray
        Reflectance of interface.

    """
    theta0 = aoi * pi / 180
    theta1 = np.arcsin(n0 / n1 * sin(theta0))

    rs01 = (n0 * cos(theta0) - n1 * cos(theta1)) / (
            n0 * cos(theta0) + n1 * cos(theta1))
    rp01 = (n1 * cos(theta0) - n0 * cos(theta1)) / (
            n1 * cos(theta0) + n0 * cos(theta1))

    if polarization == 'mixed':
        return 0.5 * (np.abs(rs01) ** 2 + np.abs(rp01) ** 2)
    elif polarization == 's':
        return np.abs(rs01) ** 2
    elif polarization == 'p':
        return np.abs(rp01) ** 2
    else:
        raise Exception('Polarization must be "mixed", "s" or "p".')


