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
