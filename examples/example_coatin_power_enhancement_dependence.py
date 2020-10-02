"""Example calculation of coating SWPR and power enhancement. Makes figure
from paper.

"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

from pvarc import thin_film_reflectance
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass
from pvarc.metrics import solar_weighted_photon_reflectance


# Set thickness and porosity values.
thickness = np.arange(0, 196, 5).astype('float')
porosity = np.arange(0, 0.51, 0.025).astype('float')
swpr = np.zeros((len(thickness), len(porosity)))
wavelength = np.linspace(200,1250,200)

# Integration limits for SWPR
swpr_wavelength_min = 400
swpr_wavelength_max = 1100

for k in tqdm(range(len(porosity))):

    index_film = refractive_index_porous_silica(wavelength, porosity[k])
    index_substrate = refractive_index_glass(wavelength)

    for j in range(len(thickness)):
        # Calculate reflectance at rough.
        reflectance = thin_film_reflectance(index_film=index_film,
                                           index_substrate=index_substrate,
                                           film_thickness=thickness[j],
                                           aoi=8,
                                           wavelength=wavelength)

        swpr[j, k] = solar_weighted_photon_reflectance(
            wavelength,
            reflectance,
            wavelength_min=swpr_wavelength_min,
            wavelength_max=swpr_wavelength_max)



power_enhancement = swpr[0, 0] - swpr


plt.figure(6, figsize=(3.7, 3))
plt.clf()
plt.contourf(thickness, porosity * 100,
             power_enhancement.transpose() * 100,
             levels=15
             )
cbar = plt.colorbar()
cbar.set_label('Nominal Power Enhancement (%)', fontsize=9)
plt.xlabel('Coating Thickness (nm)', fontsize=9)
plt.ylabel('Porosity (%)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.show()
plt.savefig('figure_power_enhancement_thickness_porosity.pdf',
            bbox_inches='tight',
            pad_inches=0)



# Make a table of optimal performance
plt.figure(11)
plt.clf()
dfm = pd.DataFrame()

x_smooth = np.linspace(thickness.min(), thickness.max(),1000)
for j in [0, 2,4,6,8, 10,12,14,16]:
    x = thickness
    y = power_enhancement[:,j]
    y2 = swpr[:,j]

    f = interp1d(x,y, 'cubic')
    f2 = interp1d(x,y2, 'cubic')

    idx_max = np.argmax(f(x_smooth))
    thickness_max_pe = x_smooth[idx_max]

    plt.plot(thickness, power_enhancement[:,j]*100)
    plt.plot(thickness_max_pe, f(x_smooth[idx_max])*100,'r.')
    dfm.loc[j,'Porosity'] = '{:.0%}'.format(porosity[j])
    dfm.loc[j, 'Max NPE (%)'] = '{:.1%}'.format(f(x_smooth).max())
    dfm.loc[j, 'Min SWPR (%)'] = '{:.1%}'.format(f2(x_smooth).min())
    dfm.loc[j,'Thickness'] = '{:.1f}'.format(thickness_max_pe)

    # print('Porosity: {}, Thickness of max PE: {}'.format(porosity[j], thickness_max_pe))
plt.xlabel('Thickness (nm)')
plt.ylabel('Power Enhancement (%)')
plt.show()


print(dfm.to_latex(index=False))
