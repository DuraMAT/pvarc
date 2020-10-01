"""Example test comparing fast explicit method and tmm method for thin film
reflectance.

toddkarin
09/10/2020
"""

import numpy as np
from time import time

from pvarc.materials import refractive_index_glass, refractive_index_porous_silica
from pvarc.tmm import thin_film_reflectance_tmm
from pvarc import thin_film_reflectance

wavelength = np.linspace(200, 1250, 2000)
index_substrate = refractive_index_glass(wavelength)
index_film = refractive_index_porous_silica(wavelength)
aoi = 8
film_thickness = 120
polarization = 'mixed'

start_time = time()
R_tmm = thin_film_reflectance_tmm(index_film=index_film,
                                  index_substrate=index_substrate,
                                  film_thickness=film_thickness,
                                  aoi=aoi,
                                  wavelength=wavelength,
                                  polarization=polarization)
time_tmm = time() - start_time
print('Elapsed time for TMM method: {:.5f} s'.format(time_tmm))
start_time = time()
R_explicit = thin_film_reflectance(index_film=index_film,
                                  index_substrate=index_substrate,
                                  film_thickness=film_thickness,
                                  aoi=aoi,
                                  wavelength=wavelength,
                                  polarization=polarization)
time_explicit = time() - start_time
print('Elapsed time for explicit method: {:.5f} s'.format(time_explicit))
print('Speed up: {:0.0f}x'.format(time_tmm/time_explicit))

max_error = np.max(np.abs(R_explicit - R_tmm))
print('Max error: {}'.format(max_error))

if max_error < 1e-10:
    print('Test passed.')

# # Plot data
# plt.figure(0)
# plt.clf()
# plt.plot(wavelength, R_explicit,
#          label='explicit')
# plt.plot(wavelength, R_explicit,
#          label='TMM')
#
# plt.show()