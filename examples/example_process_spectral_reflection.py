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

from pvarc import index_BK7, thick_slab_reflection, \
    solar_weighted_photon_reflection, fit_arc_reflection_spectrum, \
    arc_reflection_model
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

# Fit model
x, ret = fit_arc_reflection_spectrum(wavelength,
                                     reflection / 1e2,
                                     model='a',
                                     aoi=8,
                                     wavelength_min=450,
                                     wavelength_max=1000)
wavelength_extend = np.linspace(300, 1250, 1000)
reflection_fit = arc_reflection_model(wavelength_extend,**x)

# Calculate solar weighted photon reflection (SWPR) using fit
swpr = solar_weighted_photon_reflection(wavelength_extend,reflection_fit)

# Calculate SWPR for glass reference
index_glass = index_BK7(wavelength_extend)
reflection_BK7 = thick_slab_reflection('mixed', index_glass,
                                       aoi=8,
                                       wavelength=wavelength_extend)
swpr_bk7 = solar_weighted_photon_reflection(wavelength_extend, reflection_BK7)

# Calculate power enhancement due to coating.
power_enchancement = swpr_bk7 - swpr

# Plot theory.
plt.plot(wavelength_extend,
         100 * reflection_fit,
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
