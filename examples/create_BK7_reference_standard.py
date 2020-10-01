"""Example for printing a list of the refractive index of BK7 glass, useful
for performing a light reference in the spectrometer software.

"""


import numpy as np
from pvarc.materials import refractive_index_glass
from pvarc import single_interface_reflectance
wavelength = np.arange(190,1125,10)
index_glass = refractive_index_glass(wavelength,type='BK7')

reflectance = single_interface_reflectance(
    n0=1.0003,
    n1=index_glass,
    aoi=8.0,
    polarization='mixed')


print('Glass reflectance')
for k in range(len(wavelength)):
    print('{}\t{:.5f}'.format(wavelength[k],reflectance[k]))

