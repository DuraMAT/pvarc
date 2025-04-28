import numpy as np
from numpy import pi, cos, sin
import numpy as np



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
    # print('TODO: need to add checks that the calculated value is not off by pi')
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
    """
    Doesn't include the absorption!

    :param index_film:
    :param index_substrate:
    :param film_thickness:
    :param wavelength:
    :param aoi:
    :param polarization:
    :param index_air:
    :return:
    """
    return 1 - thin_film_reflectance(index_film,
                            index_substrate,
                            film_thickness,
                            wavelength,
                            aoi=aoi,
                            polarization=polarization,
                            index_air=index_air)



# import numpy as np
# from numpy import pi, sin, cos
#
# def double_film(index0, index_film1,
#                             index_film2,
#                             index_substrate,
#                             thickness_film1,
#                             thickness_film2,
#                             wavelength,
#                             aoi=8.0,
#                             polarization='mixed',
#                             ):
#     """
#     Calculate R, T, A for a two-layer thin-film stack:
#
#       air (n0) -> film1 (n1, d1) -> film2 (n2, d2) -> substrate (n3, semi-inf.)
#
#     Parameters
#     ----------
#     index0 : float or ndarray
#         Refractive index of semi-infintite layer at the top (e.g. air).
#     index_film1 : float or ndarray
#         Refractive index of the first film at the specified wavelengths.
#     index_film2 : float or ndarray
#         Refractive index of the second film at the specified wavelengths.
#     index_substrate : float or ndarray
#         Refractive index of the substrate (semi-infinite).
#     thickness_film1 : float or ndarray
#         Thickness d1 of the first film, in nm.
#     thickness_film2 : float or ndarray
#         Thickness d2 of the second film, in nm.
#     wavelength : float or ndarray
#         Wavelength in nm.
#     aoi : float or ndarray
#         Angle of incidence in degrees (in the ambient).
#     polarization : {'s', 'p', 'mixed'}
#         Polarization state. 'mixed' => average of s and p.
#
#
#     Returns
#     -------
#     R, T, A : float or ndarray
#         Reflectance, Transmittance, and Absorbance for each combination of inputs.
#         Shapes broadcast to match the inputs (e.g. wavelength, aoi, etc.).
#     """
#
#     # Compute s-polarization
#     R_s, T_s, A_s = _double_film_RT(
#         n0=index0,
#         n1=index_film1,
#         n2=index_film2,
#         n3=index_substrate,
#         d1=thickness_film1,
#         d2=thickness_film2,
#         wl=wavelength,
#         aoi_deg=aoi,
#         pol='s',
#     )
#
#     # Compute p-polarization
#     R_p, T_p, A_p = _double_film_RT(
#         n0=index0,
#         n1=index_film1,
#         n2=index_film2,
#         n3=index_substrate,
#         d1=thickness_film1,
#         d2=thickness_film2,
#         wl=wavelength,
#         aoi_deg=aoi,
#         pol='p',
#     )
#
#     if polarization == 's':
#         return R_s, T_s, A_s
#     elif polarization == 'p':
#         return R_p, T_p, A_p
#     elif polarization == 'mixed':
#         # For unpolarized light, average R and T, then A = 1 - R - T (assuming no gain).
#         R = 0.5*(R_s + R_p)
#         T = 0.5*(T_s + T_p)
#         A = 1.0 - R - T
#         return R, T, A
#     else:
#         raise ValueError("polarization must be 's', 'p', or 'mixed'.")
#
#
#
# # Fresnel reflection & transmission for each interface
# def r_t(n_i, n_j, th_i, th_j, pol):
#     if pol == 's':
#         r_ij = (n_i*cos(th_i) - n_j*cos(th_j)) / (n_i*cos(th_i) + n_j*cos(th_j))
#         t_ij = 2*n_i*cos(th_i) / (n_i*cos(th_i) + n_j*cos(th_j))
#     else:  # p-polarization
#         r_ij = (n_j*cos(th_i) - n_i*cos(th_j)) / (n_j*cos(th_i) + n_i*cos(th_j))
#         t_ij = 2*n_i*cos(th_i) / (n_j*cos(th_i) + n_i*cos(th_j))
#     return r_ij, t_ij
#
#
# # Interface matrices and propagation matrices
# def interface_matrix(r_ij, t_ij):
#     return np.array([[1./t_ij,  r_ij/t_ij],
#                      [r_ij/t_ij, 1./t_ij]], dtype=complex)
#
#
# def propagation_matrix(n_i, th_i, d_i, wl):
#     """
#     Return a (..., 2, 2) complex array, representing the propagation matrix
#     for each combination of n_i, th_i, d_i, and wl in the broadcasted shape.
#     """
#     # delta will have whatever broadcasted shape results from these operations
#     delta = (2.0 * np.pi / wl) * n_i * np.cos(th_i) * d_i
#
#     # Create an output array of shape (..., 2, 2)
#     # "shape" is the broadcast shape of delta
#     out_shape = delta.shape  # e.g. (N,) or (N_x, N_y, ...)
#     mat = np.zeros(out_shape + (2, 2), dtype=complex)
#
#     # Fill the diagonal entries
#     mat[..., 0, 0] = np.exp(-1j * delta)
#     mat[..., 1, 1] = np.exp(+1j * delta)
#
#     # Off-diagonals remain 0
#     return mat
#
# def _double_film_RT(n0, n1, n2, n3, d1, d2, wl, aoi_deg, pol='s'):
#     """
#     Returns (R, T, A) for s- or p-polarization using the characteristic-matrix approach:
#       M_total = (M_{0,1} P1) (M_{1,2} P2) (M_{2,3}),
#     where 0=ambient, 1=film1, 2=film2, 3=substrate.
#
#     Then reflection amplitude r_tot = M21 / M11 => R = |r_tot|^2,
#     transmission amplitude t_tot = 1 / M11 => T factor includes (n3 cos th3)/(n0 cos th0).
#     A = 1 - R - T.
#     """
#     aoi = np.deg2rad(aoi_deg)
#
#
#     # Indices
#     # n0 = index_air  # ambient
#     # Angles in each layer via Snell's law: n0 sin(aoi) = n_i sin(theta_i)
#     theta0 = aoi
#     theta1 = np.arcsin((n0 / n1)*sin(theta0))
#     theta2 = np.arcsin((n0 / n2)*sin(theta0))
#     theta3 = np.arcsin((n0 / n3)*sin(theta0))
#
#     r01, t01 = r_t(n0, n1, theta0, theta1, pol)
#     r12, t12 = r_t(n1, n2, theta1, theta2, pol)
#     r23, t23 = r_t(n2, n3, theta2, theta3, pol)
#
#     M_01 = interface_matrix(r01, t01)
#     P1   = propagation_matrix(n1, theta1, d1, wl)
#     M_12 = interface_matrix(r12, t12)
#     P2   = propagation_matrix(n2, theta2, d2, wl)
#     M_23 = interface_matrix(r23, t23)
#
#     # Multiply them: M_total = (M_01 P1)(M_12 P2) M_23
#     # We'll do an einsum-based approach for vectorization:
#     MP1 = np.einsum('...ij,...jk->...ik', M_01, P1)
#     MP2 = np.einsum('...ij,...jk->...ik', M_12, P2)
#     Atemp = np.einsum('...ij,...jk->...ik', MP1, MP2)
#     M_total = np.einsum('...ij,...jk->...ik', Atemp, M_23)
#
#     # Reflection amplitude
#     r_tot = M_total[...,1,0] / M_total[...,0,0]
#     R = np.abs(r_tot)**2
#
#     # Transmission amplitude
#     t_tot = 1.0 / M_total[...,0,0]
#
#     # For s- or p-pol, the same formula for T factor if we assume normal usage:
#     # T = (n3 cos(theta3) / (n0 cos(theta0))) * |t_tot|^2
#     # (works for both s and p if we define them properly).
#     T_factor = (n3*cos(theta3)) / (n0*cos(theta0))
#     T = T_factor * np.abs(t_tot)**2
#
#     # Absorbance
#     A_ = 1.0 - R - T
#
#     return R, T, A_
#



def double_film(index_superstrate,
                            index_film1,
                            index_film2,
                            index_substrate,
                            thickness_film1,
                            thickness_film2,
                            wavelength,
                            aoi=8.0,
                            polarization='mixed'):
    """
    Calculate R, T, A for a two-layer thin-film stack using the explicit
    matrix multiplication:

      superstrate (n0)
        -> film1 (n1, d1)
        -> film2 (n2, d2)
        -> substrate (n3, semi-infinite).

    All inputs can be scalars or arrays (broadcastable). The function is fully
    vectorized.

    Parameters
    ----------
    index_superstrate : float or ndarray
        Refractive index of the top superstrate medium (replaces 'index_air').
    index_film1 : float or ndarray
        Refractive index of the first thin film.
    index_film2 : float or ndarray
        Refractive index of the second thin film.
    index_substrate : float or ndarray
        Refractive index of the substrate (semi-infinite).
    thickness_film1 : float or ndarray
        Thickness (d1, nm) of the first film.
    thickness_film2 : float or ndarray
        Thickness (d2, nm) of the second film.
    wavelength : float or ndarray
        Wavelength in nm.
    aoi : float or ndarray
        Angle of incidence in degrees (in the superstrate).
    polarization : {'s', 'p', 'mixed'}
        's' => s-polarization
        'p' => p-polarization
        'mixed' => unpolarized, average of s and p

    Returns
    -------
    R, T, A : float or ndarray
        Reflectance, Transmittance, Absorbance. Same shape as broadcast of inputs.
    """
    # Compute s-polarization
    R_s, T_s, A_s = _double_film_RT(index_superstrate, index_film1, index_film2, index_substrate,
                                    thickness_film1, thickness_film2,
                                    wavelength, aoi, pol='s')
    # Compute p-polarization
    R_p, T_p, A_p = _double_film_RT(index_superstrate, index_film1, index_film2, index_substrate,
                                    thickness_film1, thickness_film2,
                                    wavelength, aoi, pol='p')

    # Combine according to polarization argument
    if polarization == 's':
        return R_s, T_s, A_s
    elif polarization == 'p':
        return R_p, T_p, A_p
    elif polarization == 'mixed':
        R = 0.5*(R_s + R_p)
        T = 0.5*(T_s + T_p)
        A = 1.0 - R - T  # or 0.5*(A_s + A_p) if you prefer direct averaging
        return R, T, A
    else:
        raise ValueError("polarization must be 's', 'p', or 'mixed'.")


def _double_film_RT(n0, n1, n2, n3,
                    d1, d2,
                    wl, aoi_deg, pol='s'):
    """
    Internal helper: returns (R, T, A) for s- or p-polarization
    using the full 2-layer matrix multiplication in an explicit way:

      M_total = (M_{0->1} * P1) * (M_{1->2} * P2) * M_{2->3}.

    n0 = superstrate, n3 = substrate (semi-infinite).
    """
    # Convert AOI to radians
    aoi = np.deg2rad(aoi_deg)

    # ---------- Snell's law for angles ----------
    theta0 = aoi
    theta1 = np.arcsin((n0 / n1) * sin(theta0))  # film1
    theta2 = np.arcsin((n0 / n2) * sin(theta0))  # film2
    theta3 = np.arcsin((n0 / n3) * sin(theta0))  # substrate

    # ---------- Fresnel Reflection/Transmission Amplitudes ----------
    # r_ij, t_ij at each interface (i->j)
    r01, t01 = _r_t(n0, n1, theta0, theta1, pol)
    r12, t12 = _r_t(n1, n2, theta1, theta2, pol)
    r23, t23 = _r_t(n2, n3, theta2, theta3, pol)

    # ---------- Build the interface matrices ----------
    # M_{0->1}, M_{1->2}, M_{2->3}
    M01_11 = 1./t01
    M01_12 = r01/t01
    M01_21 = r01/t01
    M01_22 = 1./t01

    M12_11 = 1./t12
    M12_12 = r12/t12
    M12_21 = r12/t12
    M12_22 = 1./t12

    M23_11 = 1./t23
    M23_12 = r23/t23
    M23_21 = r23/t23
    M23_22 = 1./t23

    # ---------- Build the propagation matrices ----------
    # P1, P2: each is 2×2 with diagonal = exp(± j delta).
    delta1 = (2.*pi / wl) * n1 * cos(theta1) * d1
    delta2 = (2.*pi / wl) * n2 * cos(theta2) * d2

    P1_11 = np.exp(-1j*delta1)
    P1_12 = 0.0
    P1_21 = 0.0
    P1_22 = np.exp(+1j*delta1)

    P2_11 = np.exp(-1j*delta2)
    P2_12 = 0.0
    P2_21 = 0.0
    P2_22 = np.exp(+1j*delta2)

    # Now multiply these 2×2 matrices step by step, explicitly.

    # ----------------------------------------------------
    # Step 1: M_01 * P1
    # We'll define each element of M_01P1 in a broadcast manner.
    # (M_01 is shape (...), P1 is shape (...).)
    # M_01P1_11 = M_01_11*P1_11 + M_01_12*P1_21
    # M_01P1_12 = M_01_11*P1_12 + M_01_12*P1_22
    # etc...
    # ----------------------------------------------------
    M01P1_11 = M01_11 * P1_11 + M01_12 * P1_21
    M01P1_12 = M01_11 * P1_12 + M01_12 * P1_22
    M01P1_21 = M01_21 * P1_11 + M01_22 * P1_21
    M01P1_22 = M01_21 * P1_12 + M01_22 * P1_22

    # ----------------------------------------------------
    # Step 2: (M_01P1) * M_12 => call the result M01P1M12
    # ----------------------------------------------------
    M01P1M12_11 = M01P1_11 * M12_11 + M01P1_12 * M12_21
    M01P1M12_12 = M01P1_11 * M12_12 + M01P1_12 * M12_22
    M01P1M12_21 = M01P1_21 * M12_11 + M01P1_22 * M12_21
    M01P1M12_22 = M01P1_21 * M12_12 + M01P1_22 * M12_22

    # ----------------------------------------------------
    # Step 3: M01P1M12 * P2 => call the result M01P1M12P2
    # ----------------------------------------------------
    M01P1M12P2_11 = M01P1M12_11 * P2_11 + M01P1M12_12 * P2_21
    M01P1M12P2_12 = M01P1M12_11 * P2_12 + M01P1M12_12 * P2_22
    M01P1M12P2_21 = M01P1M12_21 * P2_11 + M01P1M12_22 * P2_21
    M01P1M12P2_22 = M01P1M12_21 * P2_12 + M01P1M12_22 * P2_22

    # ----------------------------------------------------
    # Step 4: (M01P1M12P2) * M_23 => final M_total
    # ----------------------------------------------------
    M_tot_11 = M01P1M12P2_11 * M23_11 + M01P1M12P2_12 * M23_21
    M_tot_12 = M01P1M12P2_11 * M23_12 + M01P1M12P2_12 * M23_22
    M_tot_21 = M01P1M12P2_21 * M23_11 + M01P1M12P2_22 * M23_21
    M_tot_22 = M01P1M12P2_21 * M23_12 + M01P1M12P2_22 * M23_22

    # Reflection amplitude: r_tot = M21 / M11
    r_tot = M_tot_21 / M_tot_11
    R = np.abs(r_tot)**2

    # Transmission amplitude: t_tot = 1 / M11
    t_tot = 1.0 / M_tot_11

    # Transmittance factor = (n3 cos(theta3) / (n0 cos(theta0))) * |t_tot|^2
    # This accounts for the final medium’s optical impedance
    T_factor = (n3 * cos(theta3)) / (n0 * cos(theta0))
    T = T_factor * np.abs(t_tot)**2

    # Absorbance
    A_ = 1.0 - R - T
    return R, T, A_


def _r_t(n_i, n_j, th_i, th_j, pol):
    """
    Fresnel reflection & transmission amplitudes for interface (i->j).
    For s- or p- polarization. Returns (r_ij, t_ij), each shape = broadcast.
    """
    if pol == 's':
        # s-polarization
        r_ij = (n_i*cos(th_i) - n_j*cos(th_j)) / (n_i*cos(th_i) + n_j*cos(th_j))
        t_ij = 2.*n_i*cos(th_i) / (n_i*cos(th_i) + n_j*cos(th_j))
    else:
        # p-polarization
        r_ij = (n_j*cos(th_i) - n_i*cos(th_j)) / (n_j*cos(th_i) + n_i*cos(th_j))
        t_ij = 2.*n_i*cos(th_i) / (n_j*cos(th_i) + n_i*cos(th_j))
    return r_ij, t_ij



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

def thick_film_absorbance(n,thickness,wavelength,aoi,path_length_enhancement=1):
    alpha = 4 * pi * np.imag(n) / wavelength
    eff_thickness = np.real(thickness / cos(aoi/180*pi)) * path_length_enhancement
    delta = alpha * eff_thickness

    # intensity reduction (i.e. absorbance) in the cell

    return 1 - np.exp(-delta)


def cell_collection_probability(n, thickness, wavelength, aoi, minority_carrier_diffusion_length):
    """
    Calculate the probability that a minority carrier is extracted for each incident photon over the thickness
    of a solar cell for multiple sets of input parameters.

    Parameters:
        n (np.ndarray or float): Array of complex refractive indices (n + ik) or a single complex refractive index.
        thickness (np.ndarray or float): Array of cell thicknesses or a single cell thickness.
        wavelength (np.ndarray or float): Array of wavelengths or a single wavelength.
        aoi (np.ndarray or float): Array of angles of incidence in degrees or a single angle.
        minority_carrier_diffusion_length (np.ndarray or float): Array of minority carrier diffusion lengths or a single diffusion length.

    Returns:
        integrated_collection_probabilities (np.ndarray or float): Array of integrated collection probabilities or a single integrated collection probability.
    """
    # Determine the maximum length of the input arrays
    max_length = max(
        np.size(n),
        np.size(thickness),
        np.size(wavelength),
        np.size(aoi),
        np.size(minority_carrier_diffusion_length)
    )

    # Convert all inputs to arrays and repeat float inputs to match the maximum length
    if not isinstance(n, np.ndarray):
        n = np.array([n] * max_length)
    if not isinstance(thickness, np.ndarray):
        thickness = np.array([thickness] * max_length)
    if not isinstance(wavelength, np.ndarray):
        wavelength = np.array([wavelength] * max_length)
    if not isinstance(aoi, np.ndarray):
        aoi = np.array([aoi] * max_length)
    if not isinstance(minority_carrier_diffusion_length, np.ndarray):
        minority_carrier_diffusion_length = np.array([minority_carrier_diffusion_length] * max_length)

    # Ensure all input arrays have the same length
    assert len(n) == len(thickness) == len(wavelength) == len(aoi) == len(minority_carrier_diffusion_length)

    # Initialize array to store integrated collection probabilities
    integrated_collection_probabilities = np.zeros(len(n))

    for i in range(len(n)):
        # Get parameters for current iteration
        n_complex = n[i]
        k = np.imag(n_complex)
        thickness_i = thickness[i]
        wavelength_i = wavelength[i]
        minority_carrier_diffusion_length_i = minority_carrier_diffusion_length[i]

        # Calculate the absorption coefficient using the imaginary part of the refractive index
        alpha = (4 * np.pi * k) / wavelength_i  # Absorption coefficient

        # Depth range
        depth = np.linspace(0, thickness_i, 20)  # Using 500 depth points for each calculation

        # Absorption probability as a function of depth
        P_abs = 1 - np.exp(-alpha * depth)

        # Collection probability as a function of depth
        P_col = np.exp(-depth / minority_carrier_diffusion_length_i)

        # Combined probability of absorption and collection
        P_combined = P_abs * P_col

        # Calculate the integrated collection probability over the thickness
        integrated_collection_probabilities[i] = np.trapz(P_combined, depth) / thickness_i

    # Return a single value if the input was a single value, otherwise return the array
    if len(integrated_collection_probabilities) == 1:
        return integrated_collection_probabilities[0]
    else:
        return integrated_collection_probabilities



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
                        light_redirection_factor=0,
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
    theta1 = np.arcsin(np.real(index_air) / np.real(index_glass_coating) * sin(theta0))
    theta2 = np.arcsin(np.real(index_air) / np.real(index_glass) * sin(theta0))
    theta3 = np.arcsin(np.real(index_air) / np.real(index_encapsulant) * sin(theta0))
    theta4 = np.arcsin(np.real(index_air) / np.real(index_cell_coating) * sin(theta0))
    theta5 = np.arcsin(np.real(index_air) / np.real(index_cell) * sin(theta0))



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

    # Light redirected back from neighboring cell
    # R1 = T1 * (1 - T1) * light_redirection_factor * np.cos(theta0)


    # Absorbance in the glass:
    A2 = thick_film_absorbance(n=n2,thickness=thickness_glass,wavelength=wavelength,aoi=theta2*180/pi)
    # A2 = A2 / glass_transmission_improvement_factor

    # Transmittance from glass to encapsulant
    T2 = single_interface_transmittance(n0=n2, n1=n3, aoi=theta2*180/pi, polarization=polarization)

    # Absorbance in the encapsulant
    A3 = thick_film_absorbance(n=n3, thickness=thickness_encpsulant, wavelength=wavelength, aoi=theta3 * 180 / pi)

    R1 = (1 - T1) * light_redirection_factor
    # R1 = light_redirection_factor * (1 - np.cos(theta0))

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

    # A5 = cell_collection_probability(n=n5, thickness=thickness_cell, wavelength=wavelength, aoi=theta5*180/pi,
    #                                  minority_carrier_diffusion_length=300e3)


    # put it all together to get the fraction of incoming light absorbed by the cell
    ret = {'EQE': np.real( (T1 + R1) * (1-A2) * T2 *  (1-A3) * T4 * A5),
           'Isc': np.real( (T1 + R1) * (1-A2) * T2 *  (1-A3) * T4 * A5) * wavelength / 1239.8,
           'Light Entering Cell': np.real(T1 * (1-A2) * T2 *  (1-A3) * T4),
           'Transmittance Glass ARC to Glass': np.atleast_1d(T1),
           'Absorbance Glass': np.atleast_1d(A2),
           'Transmittance Glass to Encapsulant': np.atleast_1d(T2),
           'Absorbance Encapsulant': np.atleast_1d(A3),
           'Transmittance Through Cell ARC to Cell': np.atleast_1d(T4),
           'Absorbance Cell': np.atleast_1d(A5),
           }

    return ret


def path_length_enhancement(wavelength, height=35, cutoff=1070, width=20 ):
    return 1 + height / ( 1 + np.exp( -(wavelength - cutoff) / width)**0.5)





if __name__ == "__main__":

    wl = np.linspace(300, 1200, 401)  # nm
    # Suppose film1: n=1.45, d=100 nm; film2: n=2.0, d=50 nm; substrate n=3.7
    # AOI=8 deg, unpolarized
    R, T, A = double_film(
        index_superstrate=1.0003,
        index_film1=1.45,
        index_film2=2.0,
        index_substrate=3.7,
        thickness_film1=100.0,
        thickness_film2=50.0,
        wavelength=wl,
        aoi=8.0,
        polarization='mixed',
    )

    print("Shapes of R, T, A:", R.shape, T.shape, A.shape)
    print("Reflectance[0], Transmittance[0], Absorbance[0]:", R[0], T[0], A[0])

    Z = path_length_enhancement(wl, 35, 1070,20)
    print(Z)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    plt.plot(wl, Z)
    plt.ylim([0,50])
    plt.show()