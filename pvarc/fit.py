import numpy as np
from scipy.optimize import minimize, basinhopping
from pvarc.materials import refractive_index_glass, refractive_index_porous_silica
from pvarc import single_interface_reflectance, thin_film_reflectance


def arc_reflection_model(wavelength,
                         thickness=125,
                         fraction_abraded=0,
                         porosity=0.3,
                         fraction_dust=0,
                         aoi=8,
                         n0=1.0003):
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

    index_substrate = refractive_index_glass(wavelength)
    index_film = refractive_index_porous_silica(wavelength, porosity=porosity)
    glass_reflectance = single_interface_reflectance(
        n0=n0,
        n1=index_substrate,
        polarization='mixed',
        aoi=aoi)

    thin_film_R = thin_film_reflectance(index_film=index_film,
                                        index_substrate=index_substrate,
                                        film_thickness=thickness,
                                        aoi=aoi,
                                        wavelength=wavelength)

    reflectance = (1 - fraction_dust) * (
            fraction_abraded * glass_reflectance + (
            1 - fraction_abraded) * thin_film_R) + fraction_dust

    return reflectance


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
    n = refractive_index_porous_silica(wavelength_min_reflection, porosity=0.5)
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
                fixed[p] = fixed_default[p]

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
        raise Exception('model options are "TP", "TPA", "TPAD" or "TAD"')

    # Increase by a factor of 100 to improve numeric accuracy.
    reflectance = reflectance * 100

    reflectance[reflectance < 0] = 0
    reflectance[reflectance > 100] = 100

    cax = np.logical_and(wavelength > wavelength_min,
                         wavelength < wavelength_max)

    wavelength_calc = wavelength.copy()

    index_substrate = refractive_index_glass(wavelength)

    glass_reflectance_calc = single_interface_reflectance(n0=1.0003,
                                                          n1=index_substrate,
                                                          aoi=aoi,
                                                          polarization='mixed',
                                                          )

    # # Get interpolator for correct
    # if not aoi == 8:
    #     raise Exception('aoi must be 8 degrees.')

    thickness_min = 50
    thickness_max = 200
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
        index_film = refractive_index_porous_silica(wavelength=wavelength,
                                         porosity=porosity)

        index_substrate = refractive_index_glass(wavelength=wavelength)
        thin_film_R = thin_film_reflectance(
            index_film=index_film,
            index_substrate=index_substrate,
            film_thickness=thickness,
            wavelength=wavelength,
            aoi=aoi
        )
        #
        # thin_film_R = thin_film_reflection_fast(
        #     wavelength=wavelength,
        #     thickness=thickness,
        #     aoi=aoi,
        #     porosity=porosity)

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
                1 - fraction_abraded) * thin_film_R) + fraction_dust

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
                                            fraction_abraded=fixed[
                                                'fraction_abraded'],
                                            fraction_dust=fixed[
                                                'fraction_dust'],
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

