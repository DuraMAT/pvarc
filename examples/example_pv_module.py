
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


# Wavelength axis
# wavelength = np.arange(200, 1251, 1).astype('float')
wavelength = np.arange(300, 1201, 1)

aoi = np.arange(0,90,1)

index_glass_coating = refractive_index_porous_silica(wavelength, porosity=0.28)
index_glass = refractive_index_glass(wavelength)
print('index glass ', index_glass)
index_encapsulant = refractive_index_eva(wavelength)
index_cell_coating = refractive_index_SiN(wavelength)
index_cell = refractive_index_silicon(wavelength)

absorbance = np.zeros((len(aoi), len(wavelength)))
glass_arc_transmission = np.zeros((len(aoi), len(wavelength)))
light_entering_cell = np.zeros((len(aoi), len(wavelength)))

thickness_cell=300e3
cell_arc_physical_improvement_factor=5
thickness_cell_coating=70
thickness_encapsulant = 0.45e-3*1e9
thickness_glass = 2e-3*1e9
thickness_glass_coating = 140
index_air =1.0003

#
# index_air = 1
# index_glass_coating = 1.0003
# thickness_glass_coating = 1
# index_glass = 1
# thickness_glass = 1e-3*1e9
# thickness_encapsulant = 1
# index_encapsulant = 1
# thickness_cell_coating = 100
# index_cell_coating = 1
# cell_arc_physical_improvement_factor = 1

for k in range(len(aoi)):
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
                          aoi=aoi[k],
                          polarization='mixed',
                          index_air=1.0003
        )
    absorbance[k,:] = ret['EQE']
    glass_arc_transmission[k,:] = ret['Transmittance Glass ARC to Glass']
    light_entering_cell[k,:] = ret['Light Entering Cell']


ret = pv_stack_absorbance(
                    index_glass_coating=index_glass_coating,
                        index_glass=index_glass,
                        index_encapsulant=index_encapsulant,
                        index_cell_coating=index_cell_coating,
                        index_cell=index_cell,
                        thickness_glass_coating=130,
                        thickness_glass=2.0e-3*1e9,
                        thickness_encpsulant=0.45e-3*1e9,
                        thickness_cell_coating=thickness_cell_coating,
                        thickness_cell=thickness_cell,
                          wavelength=wavelength,
                        cell_arc_physical_improvement_factor=cell_arc_physical_improvement_factor,
                          aoi=0,
                          polarization='mixed',
                          index_air=1.0003
        )

iam = absorbance/absorbance[0,:]
color_list = [
    '#9400D3',  # Darker Violet
    '#4B0082',  # Indigo (kept the same as it's already a distinct deep blue)
    '#0000FF',  # Blue
    '#00FF00',  # Green
    '#CCCC00',  # Darker Yellow
    '#FF7F00',  # Orange
    '#FF0000',  # Red
    '#A52A2A'  # Brown
]
# plot each IAM function:
plt.figure(0,figsize=(4.5,3.5))
# plt.figure(0)
plt.clf()
wavelength_to_plot = [365, 430,530,625,730,850,970,1050]
idx_to_plot = [np.argmin(np.abs(wa - wavelength)) for wa in wavelength_to_plot]
for k in range(len(idx_to_plot)):

    plt.plot(aoi, 100*iam[:,idx_to_plot[k]], label='{} nm'.format(wavelength[idx_to_plot[k]]),
             color=color_list[k])
plt.xlabel('AOI (degrees)')
plt.ylabel('IAM')
# plt.title('Absorbance vs. AOI')
plt.legend(loc='lower left',fontsize=8)
plt.grid('on')
plt.ylim([95,102])
plt.xlim([0,90])
plt.show()
plt.savefig('pv_module_sim IAM.png',
            dpi=300,
            bbox_inches='tight')


# tranmission through glass arc
plt.figure(11,figsize=(4.5,3.5))
plt.clf()
for k in range(len(idx_to_plot)):
    plt.plot(aoi, 100*glass_arc_transmission[:,idx_to_plot[k]]/glass_arc_transmission[0,idx_to_plot[k]],
             label='{} nm'.format(wavelength[idx_to_plot[k]]),
                color=color_list[k])
plt.xlabel('AOI (degrees)')
plt.ylabel('Relative Transmission through Glass ARC')
plt.legend(loc='lower left',fontsize=8)
plt.grid('on')
plt.ylim([95,102])
plt.xlim([0,90])
plt.show()
plt.savefig('pv_module_sim Glass ARC Transmission.png',
            dpi=300,
            bbox_inches='tight')

# light entering cell
plt.figure(12,figsize=(4.5,3.5))
plt.clf()
for k in range(len(idx_to_plot)):
    plt.plot(aoi, 100*light_entering_cell[:,idx_to_plot[k]]/light_entering_cell[0,idx_to_plot[k]],
             label='{} nm'.format(wavelength[idx_to_plot[k]]),
                color=color_list[k])
plt.xlabel('AOI (degrees)')
plt.ylabel('Relative Light Entering Cell')
plt.legend(loc='lower left',fontsize=8)
plt.grid('on')
plt.ylim([95,102])
plt.xlim([0,90])
plt.show()
plt.savefig('pv_module_sim Light Entering Cell.png',
                dpi=300,
                bbox_inches='tight')




T1 = ret['Transmittance Glass ARC to Glass']

plt.figure(1,figsize=(4.5,3.5))
plt.clf()
# fill between 0 and EQE
plt.fill_between(wavelength,0*wavelength, ret['EQE'],label='EQE')

plt.fill_between(wavelength,ret['EQE'],ret['Transmittance Glass ARC to Glass']* \
                (1-ret['Absorbance Glass']) * \
                 ret['Transmittance Glass to Encapsulant'] * \
                (1-ret['Absorbance Encapsulant']) * \
                 ret['Transmittance Through Cell ARC to Cell'],
                 label='Rear loss'
                 )
plt.fill_between(wavelength,ret['Transmittance Glass ARC to Glass']* \
                (1-ret['Absorbance Glass']) * \
                 ret['Transmittance Glass to Encapsulant'] * \
                 (1-ret['Absorbance Encapsulant']) * \
                 ret['Transmittance Through Cell ARC to Cell'],
                ret['Transmittance Glass ARC to Glass']* \
                (1-ret['Absorbance Glass']) * \
                 ret['Transmittance Glass to Encapsulant'] * \
                 (1-ret['Absorbance Encapsulant']),
                 label='Cell ARC Reflection'
                 )

plt.fill_between(wavelength,
                ret['Transmittance Glass ARC to Glass']* \
                (1-ret['Absorbance Glass']) * \
                 ret['Transmittance Glass to Encapsulant'] * \
                 (1-ret['Absorbance Encapsulant']),
                 ret['Transmittance Glass ARC to Glass'] * \
                 (1 - ret['Absorbance Glass']) * \
                 ret['Transmittance Glass to Encapsulant'],
                 label='Absorbance Encapsulant'
                 )


plt.fill_between(wavelength,
                ret['Transmittance Glass ARC to Glass']* \
                (1-ret['Absorbance Glass']) * \
                 ret['Transmittance Glass to Encapsulant'],
                 ret['Transmittance Glass ARC to Glass'] * \
                 (1 - ret['Absorbance Glass']),
                 label='Glass/Encapsulant Reflection'
                 )
plt.fill_between(wavelength,
                 ret['Transmittance Glass ARC to Glass'] * \
                 (1 - ret['Absorbance Glass']),
                ret['Transmittance Glass ARC to Glass'],
                 label='Glass Absorbance'
                 )
plt.fill_between(wavelength,
                ret['Transmittance Glass ARC to Glass'],
                 1*wavelength,
                 label='Glass ARC'
                 )


# plt.plot(wavelength, ret['EQE'], label='EQE')
# plt.plot()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
# plt.title('Absorbance vs. Wavelength')
plt.legend(loc=[0.2,0.1],fontsize=8)
plt.grid('on')
plt.xlim([300,1200])
plt.ylim([0,1])
plt.show()
plt.savefig('pv_module_sim Loss Analysis.png',
            dpi=300,
            bbox_inches='tight')

plt.figure(2)
plt.clf()

for key in ret:
    print(key)
    if len(ret[key])==1:
        ret[key] = ret[key] * np.ones(len(wavelength))

    plt.plot(wavelength, ret[key], label=key)
# plt.plot(wavelength, ret['Transmittance Through Cell ARC to Cell'],label='Cell ARC')
# plt.plot(wavelength, ret['Transmittance Glass ARC to Glass'],label='Glass ARC')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmittance Through Cell ARC to Cell')
plt.legend()
plt.grid('on')
plt.ylim([0,1])
plt.show()

plt.savefig('pv_module_sim Components.png',
            dpi=300,
            bbox_inches='tight')