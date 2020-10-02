"""

Calculate dependence of coating reflected color on thickness and porosity.

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pvarc import thin_film_reflectance
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass
from pvarc.color import spectrum_to_rgb


# Wavelength axis
wavelength = np.arange(200, 1251, 1).astype('float')

# Choice of illuminant makes a very small change to the calculated RGB color.
illuminant = 'LED-B3'

# Scan thickness and porosity.
thickness = np.arange(0, 196, 10).astype('float')
porosity = np.arange(0, 0.51, 0.05).astype('float')



col = np.zeros((3, len(thickness), len(porosity)))
col_hex = np.empty((len(thickness), len(porosity)), dtype='object')

for k in tqdm(range(len(porosity))):

    index_film = refractive_index_porous_silica(wavelength, porosity[k])
    index_substrate = refractive_index_glass(wavelength)

    for j in range(len(thickness)):
        # Calculate reflectance at rough.
        reflection = thin_film_reflectance(index_film=index_film,
                                           index_substrate=index_substrate,
                                           film_thickness=thickness[j],
                                           aoi=8,
                                           wavelength=wavelength)
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

        # Convert to hex
        col_hex[j, k] = matplotlib.colors.to_hex(rgb_wb)

        col[:,j,k] = rgb_wb

plt.figure(5, figsize=(3.7, 3))
plt.clf()
ax = plt.axes()

dthick = thickness[1] - thickness[0]
dp = porosity[1] - porosity[0]

patches = []
for j in range(len(thickness)):
    for k in range(len(porosity)):
        # plt.plot(thickness_scan[j], porosity[k], 's',
        #          markersize=45,
        #          color=col[:,j,k])

        patch = plt.Rectangle((thickness[j], porosity[k] * 100),
                              width=dthick,
                              height=dp * 100,
                              facecolor=col[:, j, k],
                              edgecolor='None',
                              alpha=1)

        ax.add_patch(patch)


plt.xlim([0, thickness.max() + dthick])
plt.ylim([0, (porosity.max() + dp) * 100])
plt.xlabel('Coating Thickness (nm)', fontsize=9)
plt.ylabel('Porosity (%)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.show()
plt.savefig('ARC_color_thickness_porosity.pdf',
            bbox_inches='tight',
            pad_inches=0)
