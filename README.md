# pvarc
Analyze anti-reflection coating measurements on the air-glass interface of a solar module

# Introduction

Many solar modules have an anti-reflection coating (ARC) on the air-glass interface. This interface most often consists of a ~125 nm layer of porous silica. This package allows the user to simulate the reflection performance of these ARCs. It also provides convenient functions for calculating the solar-weighted photon reflectance which is a good figure-of-merit of the performance of a coating.

# Installation
To install using pip, run:
```
pip install pvarc
```

To setup a virtual environment using conda/pip, create a virtual environment:
```
conda create --name pvarc -c conda-forge python=3 numpy pandas scipy matplotlib tqdm colour-science xlrd matplotlib-scalebar
conda activate pvarc
pip install tmm pvarc
```


# Examples

## Extract coating parameters from spectral reflectance

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


## Analyze RGB images 

This example shows how to build the thickness/porosity interpolator and use it to analyze a point.

```python
"""
Example building interpolator and finding thickness, porosity and swpr
from a chromaticity coordinate.

"""
from pvarc.image import build_rgb_to_thickness_porosity_interpolator_data, \
    get_thickness_porosity_interpolator_data, calculate_thickness,\
    calculate_swpr, calculate_porosity


# Build interpolator data and save to file.
# Takes a minute, but only have to do it once!
build_rgb_to_thickness_porosity_interpolator_data()

# Inspect generated data.
df = get_thickness_porosity_interpolator_data()
print(df.head())

# Calculate thickness for a particular chromaticity coordinate. Could also put
# in an entire image.
thickness = calculate_thickness(x=0.3,y=0.2)
print('Thickness (nm): ', thickness)

# Calculate SWPR for a particular chromaticity coordinate.
swpr = calculate_swpr(x=0.3,y=0.2)
print('SWPR: ', swpr)

porosity = calculate_porosity(x=0.3,y=0.2)
print('Porosity: ', porosity)

```

Here is an example for processing an RGB image:
```python
"""
Example processing RGB microscopy image.
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvarc.image import calculate_thickness, calculate_porosity, calculate_swpr

# Filename to load
filename = '../data/example_good_coating_RGB_image.tif'

# Load image
im = plt.imread(filename)

# Inspect image
plt.figure(1)
plt.clf()
plt.imshow(im / im.max())
plt.show()

# Create chromaticity coordinates.
x = im[:, :, 0] / np.sum(im, axis=2)
y = im[:, :, 1] / np.sum(im, axis=2)

# Mask off dark areas.
mask = im.sum(axis=2) < np.percentile(im.sum(axis=2), 5)

# Calculate SWPR using chromaticity method
swpr = calculate_swpr(x, y, porosity_min=0.1, porosity_max=0.45)
thickness = calculate_thickness(x, y, porosity_min=0.1, porosity_max=0.45)
porosity = calculate_porosity(x, y, porosity_min=0.1, porosity_max=0.45)

# Set low intensity locations to nan
swpr[mask] = np.nan
thickness[mask] = np.nan
porosity[mask] = np.nan

# Calculate Statistics.
print('Mean SWPR: {:.2%}'.format(np.nanmean(swpr)))
print('Mean Thickness (nm): {:.2%}'.format(np.nanmean(thickness)))
print('Mean Porosity: {:.2%}'.format(np.nanmean(porosity)))

# Make figures.
plt.figure(22)
plt.clf()
plt.pcolormesh(swpr * 100)
cbar = plt.colorbar()
cbar.set_label('SWPR (%)')
plt.gca().invert_yaxis()
plt.show()

plt.figure(23)
plt.clf()
plt.pcolormesh(thickness)
cbar = plt.colorbar()
cbar.set_label('Thickness (nm)')
plt.gca().invert_yaxis()
plt.show()

plt.figure(24)
plt.clf()
plt.pcolormesh(porosity * 100)
cbar = plt.colorbar()
cbar.set_label('Porosity (%)')
plt.gca().invert_yaxis()
plt.show()
```