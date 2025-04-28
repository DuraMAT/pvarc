import numpy as np
import pandas as pd
import matplotlib

import pvarc.metrics as metrics
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pvarc import thin_film_reflectance, pv_stack_absorbance
from pvarc.materials import (refractive_index_porous_silica, \
    refractive_index_glass, refractive_index_silicon, refractive_index_eva, refractive_index_SiN,
                             refractive_index_amorphous_silicon)
from pvarc.color import spectrum_to_rgb

from pvarc.pv_module import PVStack
from scipy.optimize import curve_fit
from pvarc import materials

topcon = PVStack(cell_type='TOPCon')
topcon_result = topcon.calculate_absorbance()
topcon_result['IAM'] = topcon_result['EQE'] / np.transpose(np.atleast_2d(topcon_result['EQE'][:,0]))


topcon_result.keys()



index_cell = materials.refractive_index_silicon(topcon.wavelength_list)
index_cell_coating = materials.refractive_index_SiN(topcon.wavelength_list)
index_cell_imag = np.imag(index_cell)
wavelength = topcon.wavelength_list
absorption_depth = wavelength / (4 * np.pi * index_cell_imag)

absportion_depth_cell_coating = wavelength / (4 * np.pi * np.imag(index_cell_coating))


wavelength = topcon.wavelength_list
aoi = topcon.aoi_list

cell_thickness = 130e3

# EQE plot
plt.figure(0)
plt.clf()
# plot absorption depth
plt.plot(wavelength, absorption_depth, label='Absorption depth Si', color='k')
plt.plot(wavelength, absportion_depth_cell_coating, label='Absorption depth SiN', color='r')
# cel thickness
plt.plot(wavelength, cell_thickness * np.ones_like(wavelength), label='Cell thickness', color='b')
plt.grid('on')
plt.yscale('log')
plt.xlabel('Wavelength (nm)')
plt.xticks(np.arange(300,1250,100))
plt.xlim([300,1200])
plt.ylim([1,1e8])
plt.ylabel('Absorption depth (nm)')
plt.legend()
plt.show()
plt.savefig(os.path.join('figures','absorption_depth.png'),dpi=400,bbox_inches='tight')




plt.figure(1,figsize=(4,3))
plt.clf()
for thickness in [10,100,1000]:
    fraction_light_absorbed = 1 - np.exp(-thickness / absorption_depth)
    plt.plot(wavelength, fraction_light_absorbed*100, label='{} nm'.format(thickness))
plt.grid('on')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Fraction of light absorbed (%)')
plt.legend()
plt.show()
plt.savefig(os.path.join('figures','fraction_light_absorbed.png'),dpi=400,bbox_inches='tight')
