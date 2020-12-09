"""

Calculate dependence of water film reflected color on thickness and porosity.

"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvarc import thin_film_reflectance
from pvarc.materials import refractive_index_glass

# Wavelength axis
wavelength = np.arange(400, 1005, 5).astype('float')

# Scan thickness (nm) and index of film
thickness = 200.
index_film = np.array([1.325])


index_substrate = refractive_index_glass(wavelength)
reflectance = thin_film_reflectance(index_film=index_film,
                                    index_substrate=index_substrate,
                                    film_thickness=thickness,
                                    aoi=0,
                                    wavelength=wavelength)

# Make a plot
plt.figure(0)
plt.clf()

plt.plot(wavelength, reflectance*100,
         label='{} nm'.format(thickness)
         )


plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (%)')
plt.legend()
plt.xlim([400,700])
# plt.savefig('example_water_film_reflectance.pdf',
#             bbox_inches='tight',
#             )

