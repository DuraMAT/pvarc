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

    This function is fully vectorized, inputs can be float or ndarray. All
    inputs that are ndarray must have the same shape. For example,
    if an array of wavelength values with shape (1, N) is passed,
    then index_film can either be a float or an ndarray with shape (1,N).

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
    reflectance : ndarray or float

        Fractional reflectance for each value in the arrays of input
        parameters. Will have same size as input arrays.

    """

    # Calculate angles from Snell's law
    theta0 = aoi * pi / 180
    print('TODO: need to add checks that the calculated value is not off by pi')
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
        return 0.5 * np.real(Rs + Rp)
    elif polarization == 's':
        return np.real(Rs)
    elif polarization == 'p':
        return np.real(Rp)
    else:
        raise Exception('Polarization must be "mixed", "s" or "p".')


def thin_film_transmittance(index_film,
                          index_substrate,
                          film_thickness,
                          wavelength,
                          aoi=8.0,
                          polarization='mixed',
                          index_air=1.0003):
    return 1 - thin_film_reflectance(index_film,
                            index_substrate,
                            film_thickness,
                            wavelength,
                            aoi=aoi,
                            polarization=polarization,
                            index_air=index_air)


def single_interface_reflectance(n0, n1, aoi=8.0, polarization='mixed'):
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


def single_interface_transmittance(n0, n1, aoi=8.0, polarization='mixed'):
    """
    Transmittance from a single interface for a wave traveling from material n0
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

    ts01 = 2 * n0 * cos(theta0) / (n0 * cos(theta0) + n1 * cos(theta1))
    tp01 = 2 * n0 * cos(theta0) / (n1 * cos(theta0) + n0 * cos(theta1))

    T_s = np.abs(ts01) ** 2 * np.real(n1 * cos(theta1)) / np.real(n0 * cos(theta0))
    T_p = np.abs(tp01) ** 2 * np.real(n1 * cos(np.conjugate(theta1))) / np.real(n0 * cos(np.conjugate(theta0)))

    if polarization == 'mixed':
        return 0.5 * (T_s + T_p)
    elif polarization == 's':
        return T_s
    elif polarization == 'p':
        return T_p
    else:
        raise Exception('Polarization must be "mixed", "s" or "p"')

def thick_film_absorbance(n,thickness,wavelength,aoi):
    alpha = 4 * pi * np.imag(n) / wavelength
    eff_thickness = np.real(thickness / cos(aoi/180*pi))
    delta = alpha * eff_thickness

    # intensity reduction (i.e. absorbance) in the cell

    return 1 - np.exp(-delta)

def pv_stack_absorbance(index_glass_coating,
                        index_glass,
                        index_encapsulant,
                        index_cell_coating,
                        index_cell,
                        thickness_glass_coating,
                        thickness_glass,
                        thickness_encpsulant,
                        thickness_cell_coating,
                        thickness_cell,
                        wavelength,
                        cell_arc_physical_improvement_factor=2,
                          aoi=8.0,
                          polarization='mixed',
                          index_air=1.0003):
    """
    Calculate absorbance in a PV cell using a simplified one-way model.

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
    reflectance : ndarray or float

        Fractional reflectance for each value in the arrays of input
        parameters. Will have same size as input arrays.

    """
    # Layer numbering:
    # 0: air
    # 1: glass coating
    # 2: glass
    # 3: encapsulant
    # 4: cell coating
    # 5: Cell

    # Calculate angles from Snell's law
    theta0 = aoi * pi / 180
    theta1 = np.arcsin(index_air / index_glass_coating * sin(theta0))
    theta2 = np.arcsin(index_air / index_glass * sin(theta0))
    theta3 = np.arcsin(index_air / index_encapsulant * sin(theta0))
    theta4 = np.arcsin(index_air / index_cell_coating * sin(theta0))
    theta5 = np.arcsin(index_air / index_cell * sin(theta0))


    # Give index of refraction values better names.
    n0 = index_air
    n1 = index_glass_coating
    n2 = index_glass
    n3 = index_encapsulant
    n4 = index_cell_coating
    n5 = index_cell

    # Calculate transmittance through the first stack (the Glass Coating)
    T1 = thin_film_transmittance(index_film=n1,
                                 index_substrate=n2,
                                 film_thickness=thickness_glass_coating,
                                 wavelength=wavelength,
                                 aoi=aoi,
                                 polarization=polarization,
                                 index_air=n0)


    # Absorbance in the glass:
    A2 = thick_film_absorbance(n=n2,thickness=thickness_glass,wavelength=wavelength,aoi=theta2*180/pi)
    # A2 = A2 / glass_transmission_improvement_factor

    # Transmittance from glass to encapsulant
    T2 = single_interface_transmittance(n0=n2, n1=n3, aoi=theta2*180/pi, polarization=polarization)

    # Absorbance in the encapsulant
    A3 = thick_film_absorbance(n=n3, thickness=thickness_encpsulant, wavelength=wavelength, aoi=theta3 * 180 / pi)

    # Transmittance through cell ARC into the cell
    T4 = thin_film_transmittance(index_film=n4,
                                 index_substrate=n5,
                                 film_thickness=thickness_cell_coating,
                                 wavelength=wavelength,
                                 aoi=theta3/pi*180,
                                 polarization=polarization,
                                 index_air=n3)
    T4 = 1 - (1 - T4) / cell_arc_physical_improvement_factor
    A5 = thick_film_absorbance(n=n5,thickness=thickness_cell,wavelength=wavelength,aoi=theta5*180/pi)

    # put it all together to get the fraction of incoming light absorbed by the cell
    ret = {'EQE': np.real(T1 * (1-A2) * T2 *  (1-A3) * T4 * A5),
           'Light Entering Cell': np.real(T1 * (1-A2) * T2 *  (1-A3) * T4),
           'Transmittance Glass ARC to Glass': np.atleast_1d(T1),
           'Absorbance Glass': np.atleast_1d(A2),
           'Transmittance Glass to Encapsulant': np.atleast_1d(T2),
           'Absorbance Encapsulant': np.atleast_1d(A3),
           'Transmittance Through Cell ARC to Cell': np.atleast_1d(T4),
           'Absorbance Cell': np.atleast_1d(A5),
           }

    return ret

