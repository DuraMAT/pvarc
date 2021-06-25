import numpy as np
import pandas as pd
import os
# import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from tqdm import tqdm
from pvarc import thin_film_reflectance, single_interface_reflectance
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass
# from pvarc.color import spectrum_to_rgb, spectrum_to_XYZ, spectrum_to_xyY
from pvarc.metrics import solar_weighted_photon_reflectance
# from glob import glob
#
# from pvarc.oceaninsight import read_oceanview_file
#
# from colour import SpectralDistribution, CubicSplineInterpolator, Extrapolator, \
#     XYZ_to_sRGB, sd_to_XYZ, XYZ_to_xyY, XYZ_to_CIECAM02, xyY_to_XYZ, xy_to_xyY, \
#     RGB_to_XYZ, XYZ_to_xyY, \
#     Lab_to_XYZ, ILLUMINANTS, ILLUMINANTS_SDS, XYZ_to_RGB, sRGB_to_XYZ
#
# from colour import xy_to_xyY, RGB_to_XYZ, XYZ_to_xyY
# from skimage.color import rgb2xyz
from scipy.interpolate import griddata, RegularGridInterpolator


def calculate_rgb_vs_thickness_porosity(
        camera='Ximea-MC050cg_combined_labeled.csv',
        light_source='LEDW7E_A01_intensity.csv',
        optical_system='MVL23M23.csv',
        thickness_max=186,
        thickness_step=0.4,
        porosity_max=0.5,
        porosity_step=0.01,
        aoi=0):
    # Wavelength axis
    wavelength = np.arange(300, 755, 5).astype('float')
    dwavelength = wavelength[1] - wavelength[0]

    # Scan thickness and porosity.
    thickness = np.arange(0, thickness_max, thickness_step).astype('float')
    porosity = np.arange(0, porosity_max, porosity_step).astype('float')

    # Initialize arrays.
    swpr = np.zeros((len(thickness), len(porosity)))
    rgb_wb = np.zeros((len(thickness), len(porosity), 3))

    # Load Camera QE
    qe_fpath = os.path.join(os.path.dirname(__file__), 'cameras',
                            camera)
    df = pd.read_csv(qe_fpath, skiprows=2)
    dfi = pd.DataFrame({'Wavelength': wavelength})
    for k in ['Red', 'Green', 'Blue']:
        dfi[k] = np.interp(wavelength, df['Wavelength'], df[k],
                           left=0, right=0)

    # Load Illuminant spectrum
    illum_fpath = os.path.join(os.path.dirname(__file__), 'sources',
                               light_source)
    df_illum = pd.read_csv(illum_fpath, skiprows=2)
    # illuminant_sd = ILLUMINANTS_SDS[sources[s]]
    # illuminant_conv = ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant_name]

    illuminant_spectrum = np.interp(wavelength, df_illum['Wavelength'],
                                    df_illum['Intensity'], left=0, right=0)

    illuminant_spectrum_photon = illuminant_spectrum * wavelength

    # Load optical system
    optical_system_fpath = os.path.join(os.path.dirname(__file__), 'cameras',
                                        optical_system)
    df_optical_system = pd.read_csv(optical_system_fpath, skiprows=2)
    optical_system_transmission = 0.01 * np.interp(
        wavelength,
        df_optical_system['Wavelength'],
        df_optical_system['Transmission'],
        left=0, right=0)

    # Calculate RGB colors
    for k in tqdm(range(len(porosity))):

        index_film = refractive_index_porous_silica(wavelength, porosity[k])
        index_substrate = refractive_index_glass(wavelength)

        for j in range(len(thickness)):

            # Calculate reflectance
            reflectance = thin_film_reflectance(index_film=index_film,
                                                index_substrate=index_substrate,
                                                film_thickness=thickness[j],
                                                aoi=aoi,
                                                wavelength=wavelength)
            # for c in ['Blue', 'Green', 'Red']:
            #     np.sum(reflectance * illuminant_spectrum_photon * dfi[c] / 100) * dwavelength

            rgb = np.sum(
                reflectance[:, np.newaxis] * \
                optical_system_transmission[:, np.newaxis] * \
                illuminant_spectrum_photon[:, np.newaxis] * \
                np.array(dfi.iloc[:, 1:]) / 100,
                axis=0) * dwavelength

            swpr[j, k] = solar_weighted_photon_reflectance(wavelength,
                                                           reflectance)

            # Use first run through, with 0 nm thickness, (i.e. low-iron glass)
            # as a reference for white balance.
            if thickness[j] == 0:
                rgb_ref = rgb.copy()

            # White balance
            rgb_wb[j, k, :] = rgb / rgb_ref

    # x = rgb_wb[0, :, :] / np.sum(rgb_wb, axis=0)
    # y = rgb_wb[1, :, :] / np.sum(rgb_wb, axis=0)
    # z = rgb_wb[2, :, :] / np.sum(rgb_wb, axis=0)

    # t_grid, P_grid = np.meshgrid(thickness, porosity, indexing='ij')

    return thickness, porosity, rgb_wb, swpr


def build_rgb_to_thickness_porosity_interpolator_data(
        camera='Ximea-MC050cg_combined_labeled.csv',
        light_source='LEDW7E_A01_intensity.csv',
        thickness_max=500,
        thickness_step=1,
        porosity_max=0.5,
        porosity_step=0.01):
    thickness, porosity, rgb_wb, swpr = calculate_rgb_vs_thickness_porosity(
        camera=camera,
        light_source=light_source,
        thickness_max=thickness_max,
        thickness_step=thickness_step,
        porosity_max=porosity_max,
        porosity_step=porosity_step)

    x = rgb_wb[:, :, 0] / np.sum(rgb_wb, axis=2)
    y = rgb_wb[:, :, 1] / np.sum(rgb_wb, axis=2)
    t_grid, P_grid = np.meshgrid(thickness, porosity, indexing='ij')

    # Save data to file:
    df = pd.DataFrame(dict(
        x=x.flatten(),
        y=y.flatten(),
        R=rgb_wb[:, :, 0].flatten(),
        G=rgb_wb[:, :, 1].flatten(),
        B=rgb_wb[:, :, 2].flatten(),
        thickness=t_grid.flatten(),
        porosity=P_grid.flatten(),
        swpr=swpr.flatten(),
    ))

    df.to_csv(get_thickness_porosity_interpolator_filename(),
              index=False)


def get_thickness_porosity_interpolator_filename():
    return os.path.join(os.path.dirname(__file__),
                        'rgb_to_thickness_porosity_data.csv')


def get_thickness_porosity_interpolator_data():
    """
    Get data on how coating color (RGB) depends on thickness and porosity.

    Returns
    -------
    dataframe with columns

        'thickness': coating thickness in nm

        'R': Red response, white balanced to a zero thickness coating.

        'G': Green response, white balanced to a zero thickness coating.

        'B': Blue response, white balanced to a zero thickness coating.

        'x': R/(R+G+B)

        'y': G/(R+G+B)

        'swpr': Solar-weighted photon reflectance.
    """
    return pd.read_csv(get_thickness_porosity_interpolator_filename())


# Not the best way to do this, but it avoids loading data and building the
# interpolator every time the following functions are called.
if not os.path.exists(get_thickness_porosity_interpolator_filename()):
    print('Building thickness porosity interpolator data...')
    build_rgb_to_thickness_porosity_interpolator_data()

_tp_rgb = get_thickness_porosity_interpolator_data()

_thickness_list = np.unique(_tp_rgb['thickness'])
_porosity_list = np.unique(_tp_rgb['porosity'])

_tp_shape = (len(_thickness_list), len(_porosity_list))
_thickness_grid = np.array(_tp_rgb['thickness']).reshape(_tp_shape)
_porosity_grid = np.array(_tp_rgb['porosity']).reshape(_tp_shape)
_R_grid = np.array(_tp_rgb['R']).reshape(_tp_shape)
_G_grid = np.array(_tp_rgb['G']).reshape(_tp_shape)
_B_grid = np.array(_tp_rgb['B']).reshape(_tp_shape)

_R_interpolator = RegularGridInterpolator((_thickness_list, _porosity_list),
                                          _R_grid,
                                          bounds_error=False,
                                          fill_value=np.nan)
_G_interpolator = RegularGridInterpolator((_thickness_list, _porosity_list),
                                          _G_grid, bounds_error=False,
                                          fill_value=np.nan)
_B_interpolator = RegularGridInterpolator((_thickness_list, _porosity_list),
                                          _B_grid, bounds_error=False,
                                          fill_value=np.nan)


def calculate_rgb(thickness, porosity):
    coords = np.concatenate((np.atleast_1d(thickness).flatten()[:, np.newaxis],
                             np.atleast_1d(porosity).flatten()[:, np.newaxis]),
                            axis=1)

    rgb = np.concatenate((
        _R_interpolator(coords)[:, np.newaxis],
        _G_interpolator(coords)[:, np.newaxis],
        _B_interpolator(coords)[:, np.newaxis]
    ), axis=1)

    return rgb


def calculate_thickness(x, y, xy_distance_white=0.03,
                        porosity_min=0,
                        porosity_max=0.5,
                        thickness_min=0,
                        thickness_max=186,
                        ):
    """

    Parameters
    ----------
    x
    y
    xy_distance_white

    Returns
    -------

    """

    df = get_thickness_porosity_interpolator_data()
    df = df[np.logical_and.reduce(
        (df['porosity'] >= porosity_min,
         df['porosity'] <= porosity_max,
         df['thickness'] >= thickness_min,
         df['thickness'] <= thickness_max,
         ))]
    xa = np.atleast_1d(x)
    ya = np.atleast_1d(y)

    thickness_flat = griddata(np.array([df['x'], df['y']]).transpose(),
                              df['thickness'],
                              np.array(
                                  [xa.flatten(), ya.flatten()]).transpose())

    white_idx = (xa.flatten() - 1 / 3) ** 2 + (
            ya.flatten() - 1 / 3) ** 2 < xy_distance_white ** 2
    thickness_flat[white_idx] = 0

    thickness = np.reshape(thickness_flat, xa.shape)

    if np.isscalar(x) and np.isscalar(y):
        thickness = thickness[0]

    return thickness


def calculate_porosity(x, y,
                       xy_distance_white=0.03,
                       porosity_min=0,
                       porosity_max=0.5,
                       thickness_min=0,
                       thickness_max=186,
                       ):
    df = get_thickness_porosity_interpolator_data()
    df = df[np.logical_and.reduce(
        (df['porosity'] >= porosity_min,
         df['porosity'] <= porosity_max,
         df['thickness'] >= thickness_min,
         df['thickness'] <= thickness_max,
         ))]
    xa = np.atleast_1d(x)
    ya = np.atleast_1d(y)

    porosity_flat = griddata(np.array([df['x'], df['y']]).transpose(),
                             df['porosity'],
                             np.array([xa.flatten(), ya.flatten()]).transpose())

    white_idx = (xa.flatten() - 1 / 3) ** 2 + (
            ya.flatten() - 1 / 3) ** 2 < xy_distance_white ** 2
    porosity_flat[white_idx] = 0

    porosity = np.reshape(porosity_flat, xa.shape)

    if np.isscalar(x) and np.isscalar(y):
        porosity = porosity[0]

    return porosity


def calculate_swpr(x, y, xy_distance_white=0.03,
                   porosity_min=0,
                   porosity_max=0.5,
                   thickness_min=0,
                   thickness_max=186,
                   ):
    df = get_thickness_porosity_interpolator_data()
    df = df[np.logical_and.reduce(
        (df['porosity'] >= porosity_min,
         df['porosity'] <= porosity_max,
         df['thickness'] >= thickness_min,
         df['thickness'] <= thickness_max,
         ))]

    xa = np.atleast_1d(x)
    ya = np.atleast_1d(y)

    swpr_flat = griddata(np.array([df['x'], df['y']]).transpose(),
                         df['swpr'],
                         np.array([xa.flatten(), ya.flatten()]).transpose())

    white_idx = (xa.flatten() - 1 / 3) ** 2 + (
            ya.flatten() - 1 / 3) ** 2 < xy_distance_white ** 2
    swpr_flat[white_idx] = df.iloc[0]['swpr']

    swpr = np.reshape(swpr_flat, xa.shape)

    if np.isscalar(x) and np.isscalar(y):
        swpr = swpr[0]

    return swpr
