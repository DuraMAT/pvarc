import numpy as np
import pandas as pd
from tmm import coh_tmm, unpolarized_RT
from time import time
from numpy import pi, inf
import os
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import minimize, basinhopping
from tqdm import tqdm

# start_time = time()

# Inport the calculated thin film reflection data. This interpolator is used for
# the fast calculations.
cd = os.path.dirname(os.path.abspath(__file__))
thin_film_reflection_data = np.load(
    os.path.join(cd, 'thin_film_reflectance_calculated_values.npz'))

thin_film_interpolator = RegularGridInterpolator(
    (thin_film_reflection_data['thickness'],
     thin_film_reflection_data['wavelength'],
     thin_film_reflection_data['aoi'],
     thin_film_reflection_data['porosity']),
    thin_film_reflection_data['reflectance_uint16'].astype('float') *
    thin_film_reflection_data['reflectance_scale_factor'],
    bounds_error=True,
    fill_value=np.nan)

# print('Time to import interpolator: {}'.format(time() - start_time))


def index_BK7(wavelength):
    """
    Index of refraction of Schott glass from wavelength in nm.

    https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT


    Parameters
    ----------
    wavelength : numeric

        Wavelength in nm.

    Returns
    -------

    index : numeric

        Index of refraction of BK7 class.

    """
    wavelength = wavelength / 1000
    n = np.sqrt(1 + \
                (1.03961212 * wavelength ** 2) / (
                        wavelength ** 2 - 0.00600069867) + \
                (0.231792344 * wavelength ** 2) / (
                        wavelength ** 2 - 0.0200179144) + \
                (1.01046945 * wavelength ** 2) / (wavelength ** 2 - 103.560653)
                )
    return n


def index_porous_silica(wavelength, porosity=0.5):
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

        Index of refraction of BK7 class.

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
    # Reduce n to account for porosity, roughly.
    # n = n / n.mean() * average_index
    return n_total


def thick_slab_reflection(polarization, index_substrate, aoi, wavelength):
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


def thin_film_reflection(polarization, index_film, index_substrate,
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


def thin_film_reflection_fast(wavelength, thickness=120, aoi=8, porosity=0.1):
    """

    Calculates the reflection from a stack. Note an out of bound error will
    occur if the input parameters are out of bounds from the pre-calculated
    values.

    See Also : build_arc_reflection_model_interpolator_data

    Parameters
    ----------
    wavelength : ndarray
        wavelength in nm

    thickness : float
        ARC thickness in nm

    aoi : float
        aoi in degrees.

    porosity : float
        ARC porosity between 0 and 1.


    Returns
    -------

    """

    coords = np.array([np.full_like(wavelength, thickness),
                       wavelength,
                       np.full_like(wavelength, aoi),
                       np.full_like(wavelength, porosity)
                       ]).transpose()

    reflection = thin_film_interpolator(coords)

    return reflection

#
# def get_eqe(wavelength,
#             type='multi-Si'):
#     cd = os.path.dirname(os.path.abspath(__file__))
#
#     df = pd.read_csv(os.path.join(cd, 'multi_Si_eqe.csv'))
#
#     # eqe = np.interp(wavelength, df['wavelength'], df['eqe'])
#
#     f = interp1d(df['wavelength'], df['eqe'], kind='linear')
#     eqe = f(wavelength)
#
#     return eqe


def arc_reflection_model(wavelength,
                         thickness=125,
                         fraction_abraded=0,
                         porosity=0.3,
                         fraction_dust=0,
                         aoi=8):
    """
    Return the reflection values for a model of an aged ARC. The reflection
    is a linear combination of reflection from a thin film of a variable
    thickness and the reflection from BK7 glass.

    Parameters
    ----------
    wavelength : ndarray

        wavelength in nm

    thickness

        thickness of ARC in nm.

    fraction_abraded

        fraction of coating loss. 1 corresponds to 100% of the coating area
        removed and only underlying glass is present, 0 corresponds to the
        coating covering the entire sample.

    fraction_dust

        fraction of module area covered by dust with reflectivity of 1. A
        value of 0 corresponds to no dust (clean sample), 1 to full dust
        coverage (reflectivity of 1).

    aoi

        angle of incidence in degrees.

    Returns
    -------

    reflectance : ndarray

        Reflectance of sample at the values of wavelength specified.

    """

    index_glass = index_BK7(wavelength)
    index_film = index_porous_silica(wavelength, porosity=porosity)
    glass_reflectance = thick_slab_reflection('mixed', index_glass, aoi,
                                              wavelength)

    thin_film_reflectance = thin_film_reflection('mixed', index_film,
                                                 index_glass,
                                                 thickness,
                                                 aoi, wavelength)

    reflectance = (1 - fraction_dust) * (
            fraction_abraded * glass_reflectance + (
            1 - fraction_abraded) * thin_film_reflectance) + fraction_dust

    return reflectance



def build_arc_reflection_model_interpolator_data():
    """
    Calculate values of the thin film reflectance model for a variety of
    thickness, porosity, wavelength and aoi.

    File saved as 'thin_film_reflectance_calculated_values.npz'. Data is
    scaled to uint16.

    Returns
    -------

    """


    thickness = np.arange(0, 250, 2).astype('float')
    # wavelength = np.linspace(190, 1300, 300)
    wavelength = np.arange(190,1310,5).astype('float')
    porosity = np.array(
        [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0])

    index_glass = index_BK7(wavelength)

    aoi = np.array([0, 8, 30, 45, 60, 75, 85])

    thin_film_reflectance = np.zeros((len(thickness), len(wavelength),
                                      len(aoi), len(porosity))
                                     )

    for k in tqdm(range(len(thickness))):
        for j in range(len(aoi)):
            for m in range(len(porosity)):
                index_film = index_porous_silica(wavelength,
                                                       porosity=porosity[m])
                thin_film_reflectance[k, :, j, m] = thin_film_reflection(
                    'mixed',
                    index_film=index_film,
                    index_substrate=index_glass,
                    film_thickness=thickness[k],
                    aoi=aoi[j],
                    wavelength=wavelength)

    # Compress data
    reflectance_scale_factor = 1 / (2 ** 16 - 1)
    reflectance_uint16 = (
                thin_film_reflectance / reflectance_scale_factor).astype(
        'uint16')

    filename = 'thin_film_reflectance_calculated_values.npz'
    np.savez(filename,
             aoi=aoi,
             wavelength=wavelength,
             thickness=thickness,
             porosity=porosity,
             index_glass=index_glass,
             reflectance_scale_factor=reflectance_scale_factor,
             reflectance_uint16=reflectance_uint16
             )
    print('File saved as: {}'.format(filename))

    return thickness, wavelength, aoi, porosity, thin_film_reflectance


def estimate_arc_reflection_model_params(wavelength, reflectance, porosity=0.1,
                                         wavelength_min=450,
                                         wavelength_max=1000):
    # Smooth.
    # N = 5
    # reflectance = np.convolve(reflectance, np.ones((N,)), mode='same') / N

    cax = np.logical_and(wavelength > wavelength_min,
                         wavelength < wavelength_max)
    idx_min_reflection = np.argmin(reflectance[cax])

    reflection_min = reflectance[cax][idx_min_reflection]
    wavelength_min_reflection = wavelength[cax][idx_min_reflection]
    n = index_porous_silica(wavelength_min_reflection, porosity=0.5)
    thickness_estimate = 0.25 * wavelength_min_reflection / n

    # thickness_estimate = 120

    reflection_from_arc = float(
        arc_reflection_model(np.array([wavelength_min_reflection]),
                             thickness_estimate,
                             fraction_abraded=0,
                             fraction_dust=0,
                             porosity=porosity))
    reflection_from_glass = float(
        arc_reflection_model(np.array([wavelength_min_reflection]),
                             thickness_estimate,
                             fraction_abraded=1,
                             fraction_dust=0,
                             porosity=porosity))

    # print('Reflection from glass: {}'.format(reflection_from_glass))
    # print('Reflection from arc: {}'.format(reflection_from_arc))
    # print('Reflection from sample: {}'.format(reflection_min))

    fraction_abraded = (reflection_min - reflection_from_arc) / (
            reflection_from_glass - reflection_from_arc)

    # if thickness_estimate < 135:
    #     thickness_estimate = 135
    return {'thickness': thickness_estimate,
            'fraction_abraded': float(fraction_abraded),
            'fraction_dust': 0.001,
            'porosity': porosity
            }


def fit_arc_reflection_spectrum(wavelength,
                                reflectance,
                                x0=None,
                                aoi=8,
                                model='d',
                                fixed=None,
                                verbose=False,
                                method='basinhopping',
                                niter=20,
                                wavelength_min=450,
                                wavelength_max=1000):
    """
    This function fits an ARC model to the reflection spectrum. The ARC model
    is described in the function arc_reflection_model.

    Parameters
    ----------
    wavelength : ndarray

        Wavelength in nm

    reflectance : ndarray

        Fractional reflection between 0 and 1, unitless.

    x0 : dict

        Startpoint for fitting algorithm.

    aoi : float

        Angle of incidence of light

    model : str

        Different variation of the same model are available. The model is
        described under the function `arc_reflection_model`. The fixed values
        are specified in the input 'fixed'.

        'TP' - thickness and poristy are fit, fraction_abraded and fraction_dust
        set to fixed values.

        'TPA' - thickness, porosity and fraction_abraded are fit,
        fraction_dust is fixed.

        'TPAD' - thickness, porosity, fraction_abraded and fraction_dust are
        fit.

        'TAD' - thickness, fraction_abraded, fraction_dust are fit, porosity
        is a fixed value.

    fixed : dict

        Dictionary of parameters to be fixed. Default is:
           fixed = {'thickness': 125, 'fraction_abraded': 0, 'fraction_dust': 0,
             'porosity': 0.3}

     verbose : bool

        Whether to print output at each iteration.

    method : str

        Optimization method, can be 'minimize' or 'basinhopping'

    niter : int

        Number of basinhopping steps if method == 'basinhopping'

    wavelength_min : float

        Lower bound for wavelength values used in fit.

    wavelength_max : float

        Upper bound for wavelength values used in fit.

    Returns
    -------

    result : dict

        Dictionary of best fit values.

    ret

        Output of optimizer.

    """

    if np.mean(reflectance) > 1:
        print(
            'Warning: check that reflectance is a fractional value between 0 and 1.')

    if x0 == None:
        x0 = estimate_arc_reflection_model_params(wavelength, reflectance)

    fixed_default = {'thickness': 125,
                 'fraction_abraded': 0,
                 'fraction_dust': 0,
                 'porosity': 0.3}
    if fixed == None:
        fixed = fixed_default
    else:
        for p in fixed_default:
            if p not in fixed:
                fixed[p] =fixed_default[p]

    # print('x0: ', x0)

    scale = {'thickness': 0.01,
             'fraction_abraded': 1,
             'fraction_dust': 1000,
             'porosity': 1}

    if model == 'TPA':
        x0_list = [x0['thickness'] * scale['thickness'],
                   x0['fraction_abraded'] * scale['fraction_abraded'],
                   x0['porosity'] * scale['porosity']
                   ]
    elif model == 'TAD':
        x0_list = [x0['thickness'] * scale['thickness'],
                   x0['fraction_abraded'] * scale['fraction_abraded'],
                   x0['fraction_dust'] * scale['fraction_dust']
                   ]
    elif model == 'TPAD':
        x0_list = [x0['thickness'] * scale['thickness'],
                   x0['fraction_abraded'] * scale['fraction_abraded'],
                   x0['fraction_dust'] * scale['fraction_dust'],
                   x0['porosity'] * scale['porosity'],
                   ]
    elif model == 'TP':
        x0_list = [x0['thickness'] * scale['thickness'],
                   x0['porosity'] * scale['porosity'],
                   ]
    else:
        raise Exception('model must be "a" or "b" or "c"')

    # Increase by a factor of 100 to improve numeric accuracy.
    reflectance = reflectance * 100

    reflectance[reflectance < 0] = 0
    reflectance[reflectance > 100] = 100

    cax = np.logical_and(wavelength > wavelength_min,
                         wavelength < wavelength_max)

    wavelength_calc = wavelength.copy()

    index_glass = index_BK7(wavelength)

    glass_reflectance_calc = thick_slab_reflection('mixed',
                                                   index_glass, aoi,
                                                   wavelength)

    # # Get interpolator for correct
    # if not aoi == 8:
    #     raise Exception('aoi must be 8 degrees.')

    thickness_min = 50
    thickness_max = 245
    porosity_max = 0.499

    if model == 'TPA':
        bounds = [(thickness_min * scale['thickness'],
                   thickness_max * scale['thickness']),
                  (0, 1 * scale['fraction_abraded']),
                  (0, porosity_max * scale['porosity']),
                  ]
    elif model == 'TAD':
        bounds = [(thickness_min * scale['thickness'],
                   thickness_max * scale['thickness']),
                  (0, 1 * scale['fraction_abraded']),
                  (0, 1 * scale['fraction_dust']),
                  ]
    elif model == 'TPAD':
        bounds = [(thickness_min * scale['thickness'],
                   thickness_max * scale['thickness']),
                  (0, 1 * scale['fraction_abraded']),
                  (0, 1 * scale['fraction_dust']),
                  (0, porosity_max * scale['porosity']),
                  ]
    elif model == 'TP':
        bounds = [(thickness_min * scale['thickness'],
                   thickness_max * scale['thickness']),
                  (0, porosity_max * scale['porosity']),
                  ]

    def arc_model_c(wavelength, thickness, fraction_abraded, fraction_dust,
                    porosity):

        thin_film_reflectance = thin_film_reflection_fast(
            wavelength=wavelength,
            thickness=thickness,
            aoi=aoi,
            porosity=porosity)

        #
        # index_film = index_porous_silica(wavelength=wavelength,
        #                                  porosity=porosity)
        # thin_film_reflectance = thin_film_reflection(
        #     polarization='mixed',
        #     wavelength=wavelength,
        #     d_list=[np.inf, thickness, np.inf],
        #     index_film=index_film,
        #     index_substrate=index_glass,
        #     aoi=aoi)

        glass_reflectance = np.interp(wavelength, wavelength_calc,
                                      glass_reflectance_calc)

        reflectance = (1 - fraction_dust) * (
                fraction_abraded * glass_reflectance + (
                1 - fraction_abraded) * thin_film_reflectance) + fraction_dust

        return 100 * reflectance

    def arc_coating_error_function(x):

        # print('x: ',x)

        if model == 'TPA':
            thickness = x[0] / scale['thickness']
            fraction_abraded = x[1] / scale['fraction_abraded']
            porosity = x[2] / scale['porosity']
            reflectance_model = arc_model_c(wavelength, thickness,
                                            fraction_abraded=fraction_abraded,
                                            fraction_dust=fixed[
                                                'fraction_dust'],
                                            porosity=porosity)
        elif model == 'TAD':
            thickness = x[0] / scale['thickness']
            fraction_abraded = x[1] / scale['fraction_abraded']
            fraction_dust = x[2] / scale['fraction_dust']
            reflectance_model = arc_model_c(wavelength, thickness,
                                            fraction_abraded, fraction_dust,
                                            porosity=fixed['porosity'])
            # if verbose:
            #     print(
            #         'Thickness: {:03.2f}, Fraction Abraded: {:.1%}, Fraction dust: {:.1%}'.format(
            #             thickness,
            #             fraction_abraded,
            #             fraction_dust))
        elif model == 'TPAD':

            thickness = x[0] / scale['thickness']
            fraction_abraded = x[1] / scale['fraction_abraded']
            fraction_dust = x[2] / scale['fraction_dust']
            porosity = x[3] / scale['porosity']
            reflectance_model = arc_model_c(wavelength, thickness,
                                            fraction_abraded, fraction_dust,
                                            porosity)
        elif model == 'TP':

            thickness = x[0] / scale['thickness']
            porosity = x[1] / scale['porosity']
            reflectance_model = arc_model_c(wavelength, thickness,
                                            fraction_abraded=fixed['fraction_abraded'],
                                            fraction_dust=fixed['fraction_dust'],
                                            porosity=porosity)

        else:
            raise Exception('model type unknown')

        residual = np.mean(
            np.sqrt(np.abs(reflectance_model - reflectance) ** 2))

        return residual

    if method == 'minimize':
        res = minimize(arc_coating_error_function,
                       x0=x0_list,
                       options=dict(
                           # maxiter=100,
                           disp=verbose
                       ),
                       bounds=bounds
                       )
    elif method == 'basinhopping':

        def basinhopping_callback(x, f, accept):
            if model == 'TPA' and verbose:
                # print('x:', x)
                print(
                    '--\nThickness: {:03.2f}, Fraction Abraded: {:.1%}, Porosity: {:.1%}'.format(
                        x[0] / scale['thickness'],
                        x[1] / scale['fraction_abraded'],
                        x[2] / scale['porosity']
                    )
                )

        res = basinhopping(arc_coating_error_function,
                           x0=x0_list,
                           niter=niter,
                           minimizer_kwargs={'bounds': bounds},
                           disp=verbose,
                           callback=basinhopping_callback
                           )

    if model == 'TPA':
        result = {'thickness': res['x'][0] / scale['thickness'],
                  'fraction_abraded': res['x'][1] / scale['fraction_abraded'],
                  'fraction_dust': fixed['fraction_dust'],
                  'porosity': res['x'][2] / scale['porosity']}
    elif model == 'TAD':
        result = {'thickness': res['x'][0] / scale['thickness'],
                  'fraction_abraded': res['x'][1] / scale['fraction_abraded'],
                  'fraction_dust': res['x'][2] / scale['fraction_dust'],
                  'porosity': fixed['porosity']}
    elif model == 'TPAD':
        result = {'thickness': res['x'][0] / scale['thickness'],
                  'fraction_abraded': res['x'][1] / scale['fraction_abraded'],
                  'fraction_dust': res['x'][2] / scale['fraction_dust'],
                  'porosity': res['x'][3] / scale['porosity'],
                  }
    elif model == 'TP':
        result = {'thickness': res['x'][0] / scale['thickness'],
                  'fraction_abraded': fixed['fraction_abraded'],
                  'fraction_dust': fixed['fraction_dust'],
                  'porosity': res['x'][1] / scale['porosity'],
                  }

    return result, res



# def calculate_coating_performance(wavelength, reflectance,
#                                   eqe='multi-Si'):
#     """
#     Calculate the fractional increase in power by using a coating with the
#     given reflectance spectrum compared to using bare glass. The wavelength
#     should cover the entire range where EQE is non-negligible.
#
#     Parameters
#     ----------
#     wavelength
#     reflectance
#
#     Returns
#     -------
#
#     """
#
#     reflectance_glass = BK7_reflection('mixed', aoi=8, wavelength=wavelength)
#     sun = get_AM1p5_spectrum()
#
#     am1p5 = np.interp(wavelength, sun['wavelength nm'],
#                       sun['Direct+circumsolar W*m-2*nm-1'])
#
#     h = 6.626e-34
#     c = 2.998e8
#     q = 1.602e-19
#     band_gap = 1.1
#     am1p5_photon_flux = am1p5 / (h * c / (wavelength * 1e-9))
#
#     if not eqe in ['multi_Si']:
#         eqe = get_eqe(wavelength, type=eqe)
#     else:
#         raise Exception('Do not recognize eqe.')
#
#     d_wavelength = np.diff(wavelength)
#     d_wavelength = np.append(d_wavelength, d_wavelength[-1])
#
#     power_to_cell = q * band_gap * np.sum(
#         eqe * am1p5_photon_flux * (1 - reflectance) * d_wavelength)
#     power_to_cell_ref = q * band_gap * np.sum(
#         eqe * am1p5_photon_flux * (1 - reflectance_glass) * d_wavelength)
#
#     coating_performance = power_to_cell / power_to_cell_ref - 1
#
#     return coating_performance
