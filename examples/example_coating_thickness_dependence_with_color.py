"""
Example calculating coating reflection spectra and color of coating.

toddkarin
"""
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from colormath.color_objects import XYZColor, HSLColor, AdobeRGBColor, sRGBColor
from colormath.color_conversions import XYZ_to_RGB, Spectral_to_XYZ

from colormath.color_objects import SpectralColor

from pvarc import thin_film_reflectance
from pvarc.oceaninsight import read_oceanview_file
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass

# Get LED spectrum.
filename = 'relative_irradiance_iphone_X_flashlight.txt'
source = read_oceanview_file(filename)
wavelength_smooth = np.linspace(200, 1250, 200)
wavelength = np.arange(340, 830.5, 10)
source_spectrum = np.interp(wavelength, source['wavelength'], source['value'])
source_spectrum = source_spectrum / 10

# Or can use a blackbody spectrum
h = 6.26e-34
c = 3e8
kB = 1.381e-23
T = 6000
blackbody = 2 * h * c / (wavelength * 1e-9) ** 5 * 1 / (
        np.exp(h * c / (kB * T * wavelength * 1e-9)) - 1)
blackbody = blackbody / blackbody.max() * 10
# source_spectrum = blackbody


# Coating thickness to plot.
thickness_scan = np.array([0, 20, 40, 60, 80, 100, 120,
                           140, 160])
# aoi_scan = np.arange(70,-1,-10)
# aoi_scan = np.array([0,10,20,30,40,50,60,70])


for porosity in [0.15, 0.3]:

    plt.figure(0, figsize=(3.7, 3))
    plt.clf()
    ax = plt.axes()
    rect = ax.patch
    rect.set_facecolor('k')

    Rmat = np.zeros((len(wavelength), len(thickness_scan)))
    index_film = refractive_index_porous_silica(wavelength, porosity)
    index_film_smooth = refractive_index_porous_silica(wavelength_smooth,
                                                       porosity)
    index_substrate = refractive_index_glass(wavelength)
    index_substrate_smooth = refractive_index_glass(wavelength_smooth)


    def smoothit(y, N=10):
        return np.convolve(y, np.ones((N,)) / N, mode='valid')


    plt.plot(
        smoothit(source['wavelength'][1:-200]),
        smoothit(source['value'][1:-200]) / source['value'][1:-5].max() * 5,
        'w--',
        label='LED')

    for j in range(len(thickness_scan)):
        # Calculate reflectance at rough.
        Rmat[:, j] = thin_film_reflectance(index_film=index_film,
                                           index_substrate=index_substrate,
                                           film_thickness=thickness_scan[j],
                                           aoi=8,
                                           wavelength=wavelength)
        # Calculate
        R_smooth = thin_film_reflectance(index_film=index_film_smooth,
                                         index_substrate=index_substrate_smooth,
                                         film_thickness=thickness_scan[j],
                                         aoi=8,
                                         wavelength=wavelength_smooth)

        # Calculate the spectral color
        col_list = {}
        for k in range(len(wavelength)):
            if wavelength[k] >= 340 and wavelength[k] <= 830:
                col_list['spec_{:1.0f}nm'.format(wavelength[k])] = Rmat[k, j]
        spectral_color = SpectralColor(**col_list, observer='10')

        # Convert spectral color to xyz
        xyz = Spectral_to_XYZ(spectral_color,
                              illuminant_override=source_spectrum)

        # Use first run through, 0 nm thickness (i.e. low-iron glass) as a
        # reference for white balance.

        if thickness_scan[j] == 0:
            rgb_ref = XYZ_to_RGB(xyz, target_rgb=sRGBColor)

        # Convert color to rgb.
        rgb = XYZ_to_RGB(xyz, target_rgb=sRGBColor)

        # White balance
        rgb.rgb_r = rgb.rgb_r / rgb_ref.rgb_r
        rgb.rgb_g = rgb.rgb_g / rgb_ref.rgb_g
        rgb.rgb_b = rgb.rgb_b / rgb_ref.rgb_b

        # Write as rgb list.
        col = [rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b]

        plt.plot(wavelength_smooth, 100 * R_smooth,
                 label='{:2.0f} nm'.format(thickness_scan[j]),
                 color=col)

    l = plt.legend(fontsize=8,
                   loc='right',
                   facecolor='k',
                   )
    for text in l.get_texts():
        text.set_color("w")

    plt.text(0.95, 0.9, 'Porosity: {:.0%}'.format(porosity),
             color='w',
             horizontalalignment='right',
             transform=plt.gca().transAxes)
    plt.xlabel('Wavelength (nm)', fontsize=9)
    plt.ylabel('Reflectance (%)', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim([200, 1250])

    plt.show()
    plt.savefig(
        'ARC_color_dependence_thickness_LED_alternate02_porosity-{:.0f}percent.pdf'.format(
            porosity * 100),
        bbox_inches='tight',
        pad_inches=0)
