"""Example for importing an experimental spectrum, fitting to the
single-layer-coating model and finding the solar weighted photon reflectance
and power enhancement due to the coating.

Todd Karin
09/10/2020
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import time
from pvarc import single_interface_reflectance

from pvarc.metrics import solar_weighted_photon_reflection
from pvarc.fit import fit_arc_reflection_spectrum, arc_reflection_model
from pvarc.materials import refractive_index_glass
from pvarc.oceaninsight import read_oceanview_file

# Read data file
data = read_oceanview_file('example_panasonic-n330_reflection.txt')
crop = np.logical_and(data['wavelength'] > 300, data['wavelength'] < 1100)
wavelength = data['wavelength'][crop]
reflection = data['value'][crop]

# Plot data
plt.figure(0)
plt.clf()
plt.plot(wavelength, reflection,
         label='Data',
         color=[0, 0, 0.8])

# start timer for model fitting
start_time =  time()

# Fit model
x, ret = fit_arc_reflection_spectrum(wavelength,
                                     reflection / 1e2,
                                     model='TP',
                                     aoi=8,
                                     wavelength_min=450,
                                     wavelength_max=1000,
                                     fixed={'fraction_abraded':0},
                                     method='minimize')

print('Time for fit: {:.2f}s'.format( time()-start_time))

# Get the reflectance for the fitted model
wavelength_extend = np.linspace(300, 1250, 1000)
reflectance_fit = arc_reflection_model(wavelength_extend,**x)

# Calculate solar weighted photon reflection (SWPR) using fit
swpr = solar_weighted_photon_reflection(wavelength_extend,reflectance_fit)

# Calculate SWPR for glass reference
index_substrate = refractive_index_glass(wavelength_extend)
reflection_glass = single_interface_reflectance(
    n0=1.0003,
    n1=index_substrate,
    aoi=8,
    polarization='mixed')

swpr_glass = solar_weighted_photon_reflection(wavelength_extend, reflection_glass)

# Calculate power enhancement due to coating.
power_enchancement = swpr_glass - swpr

# Plot theory.
plt.plot(wavelength_extend,
         100 * reflectance_fit,
         label='Fit',
         linewidth=3,
         color=[1, 0.5, 0, 0.5], )
plt.text(450, 3, '*Best fit parameters*\n' + \
         'Thickness: {:.0f} nm\n'.format(x['thickness']) + \
         'Porosity: {:.0%}\n'.format(x['porosity']) + \
         'Fraction Abraded: {:.1%}\n'.format(x['fraction_abraded']) + \
         'SWPR (400-1100 nm): {:.1%}\n'.format(swpr) + \
         'PE (400-1100 nm): {:.1%}\n'.format(power_enchancement)
         ,
         fontsize=8,
         )
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection (%)')
plt.ylim([0, 8])
plt.xlim([300, 1250])
plt.legend()
plt.show()
plt.savefig('example_out.png',dpi=200)
