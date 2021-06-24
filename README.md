# pvarc
Analyze anti-reflection coating measurements on the air-glass interface of a solar module

# Introduction

Many solar modules have an anti-reflection coating (ARC) on the air-glass interface. This interface most often consists of a ~125 nm layer of porous silica. This package allows the user to simulate the reflection performance of these ARCs. It also provides convenient functions for calculating the solar-weighted photon reflectance which is a good figure-of-merit of the performance of a coating.

# Installation

To setup a virtual environment, create a virtual environment:
```
conda create --name pvarc -c conda-forge python=3 numpy pandas scipy matplotlib tqdm colour-science xlrd matplotlib-scalebar
pip install tmm
```

To install using pip, run:
```
pip install pvarc
```

# Example

The first example uses the ARC reflection model to generate a sythetic reflection curve. The coating parameters are then extracted using a fit to the same model. This example can be used to demonstrate the accuracy with which the true parameters can be extracted.

```python
"""Example for generating a synthetic reflection spectrum, fitting to the
single-layer-coating model and finding the solar weighted photon reflectance
and power enhancement due to the coating.

Todd Karin
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from pvarc import single_interface_reflectance
from pvarc.metrics import solar_weighted_photon_reflectance
from pvarc.fit import fit_arc_reflection_spectrum, arc_reflection_model
from pvarc.materials import refractive_index_glass
from pvarc.oceaninsight import read_oceanview_file


# Create a synthetic reflection curve based on the theoretical model. To run on
# a different dataset, simply import the data as `wavelength` and `reflection`.
wavelength = np.linspace(200, 1100, 500)
param_true = {
    'thickness': 125,
    'fraction_abraded': 0.05,
    'fraction_dust': 0.0,
    'porosity': 0.3}
reflection = arc_reflection_model(wavelength, **param_true)
reflection = reflection + np.random.normal(0, 2e-4, wavelength.shape)

# Plot data
plt.figure(0)
plt.clf()
plt.plot(wavelength, 100 * reflection,
         label='Data',
         color=[0, 0, 0.8])

# Fit model
x, ret = fit_arc_reflection_spectrum(wavelength,
                                     reflection,
                                     model='TPA',
                                     aoi=8,
                                     wavelength_min=450,
                                     wavelength_max=1000,
                                     method='basinhopping',
                                     verbose=True)
wavelength_extend = np.linspace(300, 1250, 1000)
reflection_fit = arc_reflection_model(wavelength_extend, **x)

# Calculate solar weighted photon reflection (SWPR) using fit
swpr = solar_weighted_photon_reflectance(wavelength_extend, reflection_fit)

# Calculate SWPR for glass reference
index_glass = refractive_index_glass(wavelength_extend)
reflection_BK7 = single_interface_reflectance(n0=1.0003,
                                       n1=index_glass,
                                       aoi=8)
swpr_bk7 = solar_weighted_photon_reflectance(wavelength_extend, reflection_BK7)

# Calculate power enhancement due to coating.
power_enchancement = swpr_bk7 - swpr

# Compare fit vs simulated value.
print('--\nComparison of true values vs. best fit')
for p in ['thickness', 'porosity', 'fraction_abraded']:
    print('{}.\t True: {:.2f}, Fit: {:.2f}, '.format(p, param_true[p], x[p]))

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
# plt.savefig('example_out.png',dpi=200)





```

