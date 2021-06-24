"""

Example for calculating RGB color based on thickness and porosity of
single-layer porous silica ARC.

Date Created: 06/24/2021

"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvarc.image import calculate_rgb_vs_thickness_porosity


thickness, porosity, rgb_wb, swpr = calculate_rgb_vs_thickness_porosity(
    thickness_max=600,
    thickness_step=10,
    porosity_max=1.0,
    porosity_step=0.05)

gamma = 1
rgb_wb_gamma = rgb_wb**(1/gamma)


# Plot thickness and porosity
plt.figure(0, figsize=(3.7, 3))
plt.clf()
ax = plt.axes()


dthick = thickness[1]-thickness[0]
dp = porosity[1]-porosity[0]
patches = []
for k in range(len(porosity)):
    for j in range(len(thickness)):
        patch = plt.Rectangle((thickness[j], porosity[k] * 100),
                              width=dthick,
                              height=dp * 100,
                              facecolor=rgb_wb_gamma[j,k,:],
                              edgecolor='None',
                              alpha=1)

        ax.add_patch(patch)

    # idx = np.argmin(swpr[:,k])
    # plt.plot(thickness[idx], porosity[k]*100,'r.')

plt.xlim([0, thickness.max() + dthick])
plt.ylim([0, (porosity.max() + dp) * 100])
plt.xlabel('Coating Thickness (nm)', fontsize=9)
plt.ylabel('Porosity (%)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.show()
plt.savefig('ARC_color_thickness_porosity_extended.pdf',
            bbox_inches='tight',
            pad_inches=0)


# Plot SWPR
plt.figure(1, figsize=(3.7, 3))
plt.clf()
plt.contourf(thickness, porosity*100, swpr.transpose()*100,levels=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('SWPR (%)')
plt.xlabel('Coating Thickness (nm)', fontsize=9)
plt.ylabel('Porosity (%)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.show()
plt.savefig('ARC_SWPR_extended.pdf',
            bbox_inches='tight',
            pad_inches=0)
