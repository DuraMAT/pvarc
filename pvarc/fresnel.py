import numpy as np
import time

import numpy as np

def multilayer(n_list, d_list, wavelength, aoi=0.0, polarization='mixed', coherence=None):
    """
    Compute reflectance (R), transmittance (T), and absorbance (A)
    for an arbitrary stack of finite-thickness layers, allowing each
    layer to be coherent or incoherent.

    The stack is:
        layer 0 (n0, d0) | layer 1 (n1, d1) | ... | layer N-1 (nN-1, dN-1)
    where each layer i has:
        - refractive index n_list[i],
        - thickness d_list[i].
    The code does not assume any layer is semi-infinite.

    Parameters
    ----------
    n_list : array-like, length = N
        Refractive indices for each of the N layers, including top and bottom.
        If you want a final "exit medium," add an (N+1)-th index in n_list and
        set an extra d_list entry accordingly, etc.
    d_list : array-like, length = N
        Thicknesses (nm) for each layer in n_list, including top and bottom.
        If a layer is "very thick," you can just put a large number.
    wavelength : float or ndarray
        Wavelength in nm.
    aoi : float or ndarray
        Angle of incidence in degrees in layer 0.
    polarization : {'s', 'p', 'mixed'}
        's' => s-polarization
        'p' => p-polarization
        'mixed' => average of s and p (unpolarized).
    coherence : list of bool, length = N, optional
        coherence[i] indicates whether layer i is treated as coherent (True)
        or incoherent (False). If None, all are coherent.

    Returns
    -------
    R : float or ndarray
        Overall reflectance for the entire stack.
    T : float or ndarray
        Overall transmittance for the entire stack.
    A : float or ndarray
        Overall absorbance = 1 - R - T (for real n, might be ~0 if no absorption).
    R_layers : ndarray, shape=(N,) + broadcast_shape
        Estimated reflectance "up to" each layer boundary.
    T_layers : ndarray, shape=(N,) + broadcast_shape
        Estimated transmittance "through" each layer boundary.
    A_layers : ndarray, shape=(N,) + broadcast_shape
        Estimated absorption in each layer i (very approximate if coherent).
    """

    # If user didn't specify coherence, assume everything is coherent
    if coherence is None:
        coherence = [True]*len(n_list)
    elif len(coherence) != len(n_list):
        raise ValueError("coherence must have the same length as n_list.")

    # Convert aoi to radians
    theta0 = np.deg2rad(aoi)

    # For s- and p-polarizations
    R_s, T_s, A_s, RL_s, TL_s, AL_s = _multilayer_RT(
        n_list, d_list, wavelength, theta0, pol='s', coherence=coherence
    )
    R_p, T_p, A_p, RL_p, TL_p, AL_p = _multilayer_RT(
        n_list, d_list, wavelength, theta0, pol='p', coherence=coherence
    )

    # Combine based on polarization
    if polarization == 's':
        R, T, A = R_s, T_s, A_s
        R_layers, T_layers, A_layers = RL_s, TL_s, AL_s
    elif polarization == 'p':
        R, T, A = R_p, T_p, A_p
        R_layers, T_layers, A_layers = RL_p, TL_p, AL_p
    elif polarization == 'mixed':
        R = 0.5*(R_s + R_p)
        T = 0.5*(T_s + T_p)
        # For A, we can do 1 - R - T or direct average:
        A = 1.0 - R - T
        # For per-layer arrays, let's do direct averages:
        R_layers = 0.5*(RL_s + RL_p)
        T_layers = 0.5*(TL_s + TL_p)
        A_layers = 0.5*(AL_s + AL_p)
    else:
        raise ValueError("polarization must be 's', 'p', or 'mixed'.")

    return R, T, A, R_layers, T_layers, A_layers


def _multilayer_RT(n_list, d_list, wavelength, theta0, pol='s', coherence=None):
    """
    Internal function to compute R, T, A, plus layer-by-layer breakdown,
    for a single polarization.
    """

    # Convert to numpy arrays
    # If we want each n_list[i] to be an array, do it item-by-item:
    n_list = [np.atleast_1d(x) for x in n_list]

    # Similarly for d_list if you have a mixture of scalars and arrays
    d_list = [np.atleast_1d(d) for d in d_list]

    wavelength = np.atleast_1d(wavelength)
    theta0 = np.atleast_1d(theta0)

    N = len(n_list)
    if len(d_list) != N:
        raise ValueError("d_list must have the same length as n_list.")

    # Snell's law: find angle in each layer
    theta = np.zeros((N,)+theta0.shape, dtype=complex)
    theta[0] = theta0
    for i in range(1, N):
        # n0 sin(theta0) = n_i sin(theta_i)
        sin_val = (n_list[0] * np.sin(theta0)) / n_list[i]
        theta[i] = np.arcsin(sin_val)

    # Initialize storage for layer-by-layer results
    R_layers = np.zeros((N,)+wavelength.shape, dtype=float)
    T_layers = np.zeros((N,)+wavelength.shape, dtype=float)
    A_layers = np.zeros((N,)+wavelength.shape, dtype=float)

    # Start with identity matrix
    M11 = np.ones_like(wavelength, dtype=complex)
    M12 = np.zeros_like(wavelength, dtype=complex)
    M21 = np.zeros_like(wavelength, dtype=complex)
    M22 = np.ones_like(wavelength, dtype=complex)

    # Fresnel helper
    def r_t(n_i, n_j, th_i, th_j, pol):
        if pol == 's':
            r_ij = (n_i*np.cos(th_i) - n_j*np.cos(th_j)) / \
                   (n_i*np.cos(th_i) + n_j*np.cos(th_j))
            t_ij = 2.0*n_i*np.cos(th_i) / \
                   (n_i*np.cos(th_i) + n_j*np.cos(th_j))
        else:
            # p-polarization
            r_ij = (n_j*np.cos(th_i) - n_i*np.cos(th_j)) / \
                   (n_j*np.cos(th_i) + n_i*np.cos(th_j))
            t_ij = 2.0*n_i*np.cos(th_i) / \
                   (n_j*np.cos(th_i) + n_i*np.cos(th_j))
        return r_ij, t_ij

    # We'll consider the stack as a sequence of (layer i) with thickness d_list[i],
    # followed by interface (i -> i+1). The final layer N-1 also has thickness d_list[N-1],
    # but there's no next layer beyond i=N-1 in n_list. In other words, we have N-1
    # interfaces total if there are N layers.

    # So we'll do i in range(N-1) for the interface i->i+1:
    for i in range(N - 1):
        # First, propagate through layer i if i>0 or if needed. Actually we can do it
        # in this order:
        #  1) If i>0, we've just come from interface i-1->i; otherwise, i=0 is the top.
        #  2) If layer i is coherent => add phase
        #     If layer i is incoherent => Beer–Lambert, no interference
        #  3) Then handle interface i->(i+1).
        # However, you might prefer a different order. We'll do it as in your prior code.

        # For the very top layer i=0, there's no previous interface. But we can still do
        # the “propagation” if thickness>0.
        if coherence[i]:
            # Coherent => multiply by e^(±j delta)
            delta = (2.0*np.pi / wavelength) * n_list[i] * d_list[i] * np.cos(theta[i])
            exp_neg = np.exp(-1j * delta)
            exp_pos = np.exp(+1j * delta)
            M11, M12, M21, M22 = (M11*exp_neg, M12*exp_pos,
                                  M21*exp_neg, M22*exp_pos)
        else:
            # Incoherent => Beer–Lambert
            alpha = 4.0*np.pi*np.imag(n_list[i]) / wavelength  # absorption coefficient
            angle_factor = 1.0 / np.cos(theta[i])              # path length factor
            attenuation = np.exp(-0.5 * alpha * d_list[i] * angle_factor)
            M11, M12, M21, M22 = (M11*attenuation, M12*attenuation,
                                  M21*attenuation, M22*attenuation)
            # Estimate absorption in this layer:
            # Approx. fraction lost: 1 - exp(-alpha*d/cos(theta)).
            # "Incident intensity" is T_layers[i], maybe. But since we haven't done
            # the interface yet, let's store a partial guess.
            # We'll do that *after* we compute reflection/transmission at the next interface.

        # Now do interface (i -> i+1)
        r_ij, t_ij = r_t(n_list[i], n_list[i+1], theta[i], theta[i+1], pol)

        # Multiply interface matrix
        M_int_11 = 1.0 / t_ij
        M_int_12 = r_ij / t_ij
        M_int_21 = r_ij / t_ij
        M_int_22 = 1.0 / t_ij

        A11 = M11*M_int_11 + M12*M_int_21
        A12 = M11*M_int_12 + M12*M_int_22
        A21 = M21*M_int_11 + M22*M_int_21
        A22 = M21*M_int_12 + M22*M_int_22
        M11, M12, M21, M22 = A11, A12, A21, A22

        # "Instantaneous" reflectance & transmittance if we ended here
        r_tot_here = M21 / M11
        R_here = np.abs(r_tot_here)**2
        t_tot_here = 1.0 / M11

        # For "transmitted" fraction, we assume the next layer i+1 is the "exit" for this step:
        n0 = n_list[0]
        th0 = theta[0]
        n_next = n_list[i+1]
        th_next = theta[i+1]
        T_here = (n_next * np.cos(th_next) / (n0 * np.cos(th0))) * np.abs(t_tot_here)**2

        # Store partial results
        R_layers[i+1] = R_here
        T_layers[i+1] = T_here
        # By default for coherent, set A_layers=0 here.
        # If we want a better breakdown, we'd do more advanced tracking:
        A_layers[i+1] = 0.0

        # If i == N-2, that means we've just handled the last interface. So there's no next interface.
        # Otherwise, we proceed to the next iteration, which will do the next layer's propagation, etc.

    # Finally, we have also to propagate through the *last layer* (index N-1) if needed:
    last = N - 1
    if coherence[last]:
        # Coherent
        delta = (2.0*np.pi / wavelength) * n_list[last] * d_list[last] * np.cos(theta[last])
        exp_neg = np.exp(-1j * delta)
        exp_pos = np.exp(+1j * delta)
        M11, M12, M21, M22 = (M11*exp_neg, M12*exp_pos, M21*exp_neg, M22*exp_pos)
    else:
        # Incoherent
        alpha = 4.0*np.pi*np.imag(n_list[last]) / wavelength
        angle_factor = 1.0 / np.cos(theta[last])
        attenuation = np.exp(-0.5 * alpha * d_list[last] * angle_factor)
        M11, M12, M21, M22 = (M11*attenuation, M12*attenuation,
                              M21*attenuation, M22*attenuation)
        # Estimate absorption in last layer. We'll be approximate again:
        # let's use the last T_layers[last], or do a final pass after reflection?
        # For simplicity:
        A_layers[last] = T_layers[last] * (1.0 - np.exp(-alpha*d_list[last]*angle_factor))

    # Now, the *overall* reflection & transmission from the entire stack
    r_tot = M21 / M11
    R_final = np.abs(r_tot)**2
    t_tot = 1.0 / M11
    # For "transmitted" fraction, we treat the final exit as if there's "nothing" beyond the last layer,
    # or you can define an n_exit if there's an external medium. Let's define n_exit=1 and angle=0 for simplicity,
    # or assume that "the last layer leads to air." That means we do need an interface if that's the real scenario...
    #
    # Actually, let's just do a naive approach: If there's an external medium n_ext,
    # you'd define that as layer N, etc.
    # We'll do a minimal approach: the fraction that "exits" the last layer is:
    n0 = n_list[0]
    th0 = theta[0]
    n_last = n_list[last]
    th_last = theta[last]
    T_final = (n_last*np.cos(th_last)/(n0*np.cos(th0))) * np.abs(t_tot)**2

    A_final = 1.0 - R_final - T_final

    return R_final, T_final, A_final, R_layers, T_layers, A_layers

# Suppose `multilayer` is imported from our code:
# from my_multilayer_code import multilayer

def test_speed_and_accuracy_with_tmm():
    """
    Compare speed and accuracy of the 'multilayer' function vs. 'tmm' for 10k wavelengths.
    We'll compute R, T, A over a range of wavelengths, measure time, and compare results.
    """

    # ---------------------------------------------------------
    # 1) Define the stack
    # ---------------------------------------------------------
    # We'll do a two-layer stack again for simplicity:
    #   air (n0=1.0) | SiO2 layer (n1=1.45) | Si substrate (n2=3.7)
    # thickness of 100 nm
    n_air = 1.0
    n_sio2 = 1.45
    n_si = 3.7
    thickness_nm = 130.0
    n_list = [n_air, n_sio2, n_si]
    d_list = [1000, thickness_nm, 1000]  # single interior layer

    # angle of incidence
    aoi_deg = 10.0

    # ---------------------------------------------------------
    # 2) Define wavelength range
    # ---------------------------------------------------------
    num_points = 10_000
    wavelength_start_nm = 400.0
    wavelength_end_nm = 800.0
    wavelengths_nm = np.linspace(wavelength_start_nm, wavelength_end_nm, num_points)

    # We'll do s-polarization (you could also test p or mixed).
    pol = 's'

    # ---------------------------------------------------------
    # 3) Time & run our `multilayer` code
    # ---------------------------------------------------------
    start_time = time.time()
    R_my, T_my, A_my, R_layers, T_layers, A_layers = multilayer(
        n_list=n_list,
        d_list=d_list,
        wavelength=wavelengths_nm,
        aoi=aoi_deg,
        polarization=pol
    )
    elapsed_ours = time.time() - start_time

    # ---------------------------------------------------------
    # 4) Time & run tmm code
    # ---------------------------------------------------------
    # TMM uses consistent units; let's do microns:
    #  - convert nm to microns
    wavelengths_um = wavelengths_nm * 1e-3

    # Prebuild arrays for R, T, A from tmm
    R_tmm = np.zeros_like(wavelengths_nm)
    T_tmm = np.zeros_like(wavelengths_nm)
    A_tmm = np.zeros_like(wavelengths_nm)

    d_tmm = [np.inf, thickness_nm * 1e-3, np.inf]  # layer thicknesses in microns
    n_tmm = [n_air, n_sio2, n_si]

    start_time = time.time()
    for i, wl_um in enumerate(wavelengths_um):
        result = coh_tmm(
            pol=pol,
            n_list=n_tmm,
            d_list=d_tmm,
            th_0=np.deg2rad(aoi_deg),
            lam_vac=wl_um
        )
        R_tmm[i] = result['R']
        T_tmm[i] = result['T']
        A_tmm[i] = 1 - R_tmm[i] - T_tmm[i]
    elapsed_tmm = time.time() - start_time

    # ---------------------------------------------------------
    # 5) Compare times
    # ---------------------------------------------------------
    print(f"\nNumber of points: {num_points}")
    print(f"multilayer function time: {elapsed_ours:.3f} s")
    print(f"tmm loop time:            {elapsed_tmm:.3f} s")

    # ---------------------------------------------------------
    # 6) Compare accuracy
    # ---------------------------------------------------------
    # We'll check a handful of points or the entire array using isclose
    # (you can choose a tolerance that makes sense for your problem).
    diff_R = np.max(np.abs(R_my - R_tmm))
    diff_T = np.max(np.abs(T_my - T_tmm))
    diff_A = np.max(np.abs(A_my - A_tmm))

    print("\nMax difference in Reflectance:   ", diff_R)
    print("Max difference in Transmittance: ", diff_T)
    print("Max difference in Absorbance:    ", diff_A)

    # If you want to enforce an accuracy threshold:
    assert diff_R < 1e-6, f"Reflectance mismatch too large (max diff={diff_R:.2e})"
    assert diff_T < 1e-6, f"Transmittance mismatch too large (max diff={diff_T:.2e})"
    assert diff_A < 1e-6, f"Absorbance mismatch too large (max diff={diff_A:.2e})"


if __name__ == "__main__":
    test_speed_and_accuracy_with_tmm()

if __name__ == "__main__":
    # Example: air | (SiO2 layer) | Si substrate
    n_air = 1.0
    n_sio2 = 1.45 + 0.0j
    n_si = 3.7 + 0.1j
    n_list = [n_air, n_sio2, n_si]
    d_list = [5000, 100.0, 1000]  # nm

    # Suppose we treat the SiO2 layer as coherent, and the substrate as incoherent (doesn't matter much)
    # The first element in coherence is for n_list[0] (ambient),
    # second for n_list[1], third for n_list[2], etc.
    coherence_flags = [True, True, False]

    # Let's do a range of wavelengths
    wavelengths = np.linspace(400, 800, 5)  # just 5 points
    aoi_deg = 10.0

    R, T, A, R_layers, T_layers, A_layers = multilayer(
        n_list, d_list, wavelengths, aoi_deg,
        polarization='mixed',
        coherence=coherence_flags
    )

    print("Wavelengths (nm):", wavelengths)
    print("Reflectance =", R)
    print("Transmittance =", T)
    print("Absorbance =", A)
    print("R_layers =", R_layers)
    print("T_layers =", T_layers)
    print("A_layers =", A_layers)



# if __name__ == "__main__":
#     # Quick run (not using pytest, just a direct call)
#     test_compare_with_tmm()
#
# if __name__ == '__main__':
#     # Example usage:
#     # Air (index ~1.0003),
#     # one layer of SiO2 (thickness 100 nm),
#     # then a substrate of Si (semi-infinite).
#     n_air = 1.0003
#     n_sio2 = 1.46
#     n_si = 3.7  # example
#
#     n_list = [n_air, n_sio2, n_si]  # length 3
#     d_list = [100.0]  # length 1 (only 1 interior layer)
#
#     wls = np.linspace(400, 800, 401)  # nm
#     aoi_deg = 8.0
#
#     R, T, A = multilayer(n_list, d_list, wls, aoi=aoi_deg, polarization='mixed')
#
#     # R, T, A are arrays of length 401
#     print("Reflectance:   ", R)
#     print("Transmittance: ", T)
#     print("Absorbance:    ", A)
