"""
Example of a PV module simulation using the pvarc package.

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pvarc import thin_film_reflectance, pv_stack_absorbance
from pvarc.materials import refractive_index_porous_silica, \
    refractive_index_glass, refractive_index_silicon, refractive_index_eva, refractive_index_SiN
from pvarc.color import spectrum_to_rgb


wavelength = np.array([800,500,500])
aoi = np.array([0,10,20])


index_glass_coating = refractive_index_porous_silica(wavelength, porosity=0.28)
index_glass = refractive_index_glass(wavelength)
print('index glass ', index_glass)
index_encapsulant = refractive_index_eva(wavelength)
index_cell_coating = refractive_index_SiN(wavelength)
index_cell = refractive_index_silicon(wavelength)

thickness_cell=300e3
cell_arc_physical_improvement_factor=7
thickness_cell_coating=70
thickness_encapsulant = 0.45e-3*1e9
thickness_glass = 2e-3*1e9
thickness_glass_coating = 140
index_air =1.0003



ret = pv_stack_absorbance(
                index_glass_coating=index_glass_coating,
                    index_glass=index_glass,
                    index_encapsulant=index_encapsulant,
                    index_cell_coating=index_cell_coating,
                    index_cell=index_cell,
                    thickness_glass_coating=thickness_glass_coating,
                    thickness_glass=thickness_glass,
                    thickness_encpsulant=thickness_encapsulant,
                    thickness_cell_coating=thickness_cell_coating,
                    thickness_cell=thickness_cell,
                    wavelength=wavelength,
                    cell_arc_physical_improvement_factor=cell_arc_physical_improvement_factor,
                      aoi=aoi,
                      polarization='mixed',
                      index_air=1.0003
    )

print(ret)
print('Response: ', ret['Isc'])