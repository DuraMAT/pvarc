"""
Example calculation of coating reflection spectrum and color of coating.

toddkarin
"""
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvarc import thin_film_reflectance
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass
from pvarc.color import spectrum_to_rgb

# Wavelength axis
wavelength = np.arange(200, 1251, 1).astype('float')

# Coating thickness to plot.
thickness = np.array([0, 20, 40, 60, 80, 100, 120,
                      140, 160])

# Choice of illuminant makes a very small change to the calculated RGB color.
illuminant = 'LED-B3'

for porosity in [0.15, 0.3]:

    plt.figure(0, figsize=(3.7, 3))
    plt.clf()
    ax = plt.axes()
    rect = ax.patch
    rect.set_facecolor('k')

    index_film = refractive_index_porous_silica(wavelength, porosity)
    index_substrate = refractive_index_glass(wavelength)


    for j in range(len(thickness)):
        # Calculate reflectance
        reflectance = thin_film_reflectance(index_film=index_film,
                                           index_substrate=index_substrate,
                                           film_thickness=thickness[j],
                                           aoi=8,
                                           wavelength=wavelength)

        rgb = spectrum_to_rgb(wavelength, reflectance,
                              illuminant=illuminant)

        # Use first run through, with 0 nm thickness, (i.e. low-iron glass)
        # as a reference for white balance.
        if thickness[j] == 0:
            rgb_ref = rgb.copy()

        # White balance
        rgb_wb = rgb / rgb_ref

        # Clamp
        rgb_wb[rgb_wb >= 1] = 1
        rgb_wb[rgb_wb < 0] = 0

        if thickness[j] == 80:
            print(rgb_wb)

        plt.plot(wavelength, 100 * reflectance,
                 label='{:2.0f} nm'.format(thickness[j]),
                 color=rgb_wb
                 )

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
        'ARC_color_dependence_porosity-{:.0f}percent.pdf'.format(
            porosity * 100),
        bbox_inches='tight',
        pad_inches=0)
