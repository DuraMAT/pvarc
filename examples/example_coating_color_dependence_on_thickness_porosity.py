import numpy as np
import pvlib
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import parse

import tmm

from colormath import color_objects
from colormath.color_objects import XYZColor, HSLColor, AdobeRGBColor, sRGBColor
from colormath.color_conversions import convert_color, XYZ_to_RGB, \
    Spectral_to_XYZ, RGB_to_HSV, XYZ_to_RGB

from colormath.color_objects import SpectralColor

from numpy import pi, inf

import pvarc
from pvarc.oceaninsight import read_oceanview_file

from pvarc import thin_film_reflectance
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass
from pvarc.metrics import solar_weighted_photon_reflectance
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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
# thickness_scan = np.array([0, 20, 40, 60, 80, 100, 120,
#                            140, 160])

thickness_scan = np.arange(0, 196, 10).astype('float')
porosity = np.arange(0, 0.51, 0.05).astype('float')

col = np.zeros((3, len(thickness_scan), len(porosity)))
col_hex = np.empty((len(thickness_scan), len(porosity)), dtype='object')
col_transmission_hex = np.empty((len(thickness_scan), len(porosity)),
                                dtype='object')
swpr = np.zeros((len(thickness_scan), len(porosity)))
swpr_Si = np.zeros((len(thickness_scan), len(porosity)))
target_rgb = sRGBColor
# col_r = np.zeros((len(porosity), len(thickness_scan)))
#
# col_g = np.zeros((len(porosity), len(thickness_scan)))
# col_b = np.zeros((len(porosity), len(thickness_scan)))
swpr_wavelength_min = 400
swpr_wavelength_max = 1100
for k in tqdm(range(len(porosity))):

    index_film = refractive_index_porous_silica(wavelength, porosity[k])
    index_film_smooth = refractive_index_porous_silica(wavelength_smooth,
                                                       porosity[k])
    index_substrate = refractive_index_glass(wavelength)
    index_substrate_smooth = refractive_index_glass(wavelength_smooth)

    for j in range(len(thickness_scan)):
        # Calculate reflectance at rough.
        reflection = thin_film_reflectance(index_film=index_film,
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

        swpr[j, k] = solar_weighted_photon_reflectance(
            wavelength_smooth,
            R_smooth,
            wavelength_min=swpr_wavelength_min,
            wavelength_max=swpr_wavelength_max)

        # Calculate the spectral color
        col_list = {}
        for w in range(len(wavelength)):
            if wavelength[w] >= 340 and wavelength[w] <= 830:
                col_list['spec_{:1.0f}nm'.format(wavelength[w])] = reflection[w]

        spectral_color = SpectralColor(**col_list, observer='10')

        # Convert spectral color to xyz
        xyz = Spectral_to_XYZ(spectral_color,
                              illuminant_override=source_spectrum)

        # Use first run through, 0 nm thickness (i.e. pure glass) as a reference for white balance.
        if thickness_scan[j] == 0:
            rgb_ref = XYZ_to_RGB(xyz, target_rgb=target_rgb)

        # Convert color to rgb.
        rgb = XYZ_to_RGB(xyz, target_rgb=target_rgb)

        # White balance
        norm = 1.0
        rgb.rgb_r = rgb.rgb_r / rgb_ref.rgb_r / norm
        rgb.rgb_g = rgb.rgb_g / rgb_ref.rgb_g / norm
        rgb.rgb_b = rgb.rgb_b / rgb_ref.rgb_b / norm

        gamma = 1

        # Write as rgb list.
        col[0, j, k] = rgb.clamped_rgb_r ** gamma
        col[1, j, k] = rgb.clamped_rgb_g ** gamma
        col[2, j, k] = rgb.clamped_rgb_b ** gamma

        # Convert to hex
        col_hex[j, k] = matplotlib.colors.to_hex(col[:, j, k])
        col_transmission_hex[j, k] = matplotlib.colors.to_hex(1 - col[:, j, k])

plt.figure(5, figsize=(3.7, 3))
plt.clf()
ax = plt.axes()
rect = ax.patch
rect.set_facecolor('k')

dthick = thickness_scan[1] - thickness_scan[0]
dp = porosity[1] - porosity[0]

patches = []
for j in range(len(thickness_scan)):
    for k in range(len(porosity)):
        # plt.plot(thickness_scan[j], porosity[k], 's',
        #          markersize=45,
        #          color=col[:,j,k])

        patch = plt.Rectangle((thickness_scan[j], porosity[k] * 100),
                              width=dthick,
                              height=dp * 100,
                              facecolor=col[:, j, k],
                              edgecolor='None',
                              alpha=1)
        print(patch)
        ax.add_patch(patch)

plt.show()
plt.xlim([0, thickness_scan.max() + dthick])
plt.ylim([0, (porosity.max() + dp) * 100])

#
# l = plt.legend(fontsize=8,
#                loc='right',
#                facecolor='k',
#                )
# for text in l.get_texts():
#     text.set_color("w")

#
#
# plt.text(0.95,0.9, 'Porosity: {:.0%}'.format(porosity),
#          color='w',
#          horizontalalignment='right',
#          transform=plt.gca().transAxes)
plt.xlabel('Coating Thickness (nm)', fontsize=9)
plt.ylabel('Porosity (%)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
# plt.xlim([200, 1250])
#
# plt.show()
plt.savefig('tmm_color_thickness_porosity_gamma-{:.2f}.pdf'.format(gamma),
            bbox_inches='tight',
            pad_inches=0)

power_enhancement = swpr[0, 0] - swpr
power_enhancement_Si = swpr_Si[0, 0] - swpr_Si

plt.figure(6, figsize=(3.7, 3))
plt.clf()
plt.contourf(thickness_scan, porosity * 100,
             power_enhancement_Si.transpose() * 100,
             levels=15
             )
cbar = plt.colorbar()
cbar.set_label('Nominal Power Enhancement (%)', fontsize=9)
plt.xlabel('Coating Thickness (nm)', fontsize=9)
plt.ylabel('Porosity (%)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.show()
plt.savefig('tmm_thickness_porosity_PE_400-1100-nm.pdf',
            bbox_inches='tight',
            pad_inches=0)

color_file = 'coating_color_vs_thickness_porosity.txt'
porosity_text = ['{:.0%}'.format(p) for p in porosity]
thickness_text = ['{:.0f} nm'.format(p) for p in thickness_scan]
df = pd.DataFrame(data=col_hex, columns=porosity_text, index=thickness_text)
df_transmission = pd.DataFrame(data=col_transmission_hex, columns=porosity_text,
                               index=thickness_text)
print(df)
print(df_transmission)


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(
        int(value[i:i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


col_to_plot = hex_to_rgb('a2b4ef')

1 - np.array(col_to_plot)

power_enhancement_Si.max() - power_enhancement.max()

#
# # Make a table of optimal performance
# from scipy.interpolate import interp1d
# plt.figure(11)
# plt.clf()
# dfm = pd.DataFrame()
#
# x_smooth = np.linspace(thickness_scan.min(), thickness_scan.max(),1000)
# for j in [0, 2,4,6,8, 10,12,14,16]:
#     x = thickness_scan
#     y = power_enhancement_Si[:,j]
#     y2 = swpr_Si[:,j]
#
#     f = interp1d(x,y, 'cubic')
#     f2 = interp1d(x,y2, 'cubic')
#
#     idx_max = np.argmax(f(x_smooth))
#     thickness_max_pe = x_smooth[idx_max]
#
#     plt.plot(thickness_scan, power_enhancement[:,j])
#
#     # idx_max = np.argmax(power_enhancement[:,j])
#     plt.plot(thickness_max_pe, f2(x_smooth[idx_max]),'r.')
#     dfm.loc[j,'Porosity'] = '{:.0%}'.format(porosity[j])
#     dfm.loc[j, 'Max NPE (%)'] = '{:.1%}'.format(f(x_smooth).max())
#     dfm.loc[j, 'Min SWPR (%)'] = '{:.1%}'.format(f2(x_smooth).min())
#     dfm.loc[j,'Thickness'] = '{:.1f}'.format(thickness_max_pe)
#
#     # print('Porosity: {}, Thickness of max PE: {}'.format(porosity[j], thickness_max_pe))
# plt.show()
#
#
# print(dfm.to_latex(index=False))
