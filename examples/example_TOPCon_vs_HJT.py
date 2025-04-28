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


hjt = PVStack(cell_type='HJT')
# hjt.create_wavelength_aoi_grid(aoi_list=np.array([0]))
hjt_result = hjt.calculate_absorbance()
hjt_result['IAM'] = hjt_result['EQE'] / np.transpose(np.atleast_2d(hjt_result['EQE'][:,0]))

topcon = PVStack(cell_type='TOPCon')
topcon_result = topcon.calculate_absorbance()
topcon_result['IAM'] = topcon_result['EQE'] / np.transpose(np.atleast_2d(topcon_result['EQE'][:,0]))


wavelength = topcon.wavelength_list
aoi = topcon.aoi_list

# EQE plot
plt.figure(0,figsize=(3.5,3))
plt.clf()
plt.plot(wavelength, hjt_result['EQE'][:,0], label='HJT')
plt.plot(wavelength, topcon_result['EQE'][:,0], label='TOPCon')
# plt.plot(wavelength, hjt_result['Light Entering Cell'][:,0],'--', label='HJT - Entering Cell')
# plt.plot(wavelength, topcon_result['Light Entering Cell'][:,0],'--', label='TOPCon - Entering Cell')
# plt.plot(wavelength, topcon_result['Absorbance Cell Coating'][:,0],'--', label='TOPCon - Absorbance Cell Coating')
# plt.plot(wavelength, hjt_result['Absorbance Cell Coating'][:,0],'--', label='HJT - Absorbance Cell Coating')
# plt.plot(wavelength, topcon_result['Reflectance Cell Coating'][:,0],'--', label='TOPCon - Reflectance Cell Coating')
# plt.plot(wavelength, hjt_result['Reflectance Cell Coating'][:,0],'--', label='HJT - Reflectance Cell Coating')
plt.xlabel('Wavelength (nm)',fontsize=8)
plt.ylabel('EQE',fontsize=8)
plt.show()
plt.legend(fontsize=7)
plt.grid('on')

plt.savefig(os.path.join('figures','eqe_comparison.png'),
            dpi=450,
            bbox_inches='tight')



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
# IAM plot
plt.figure(1)
plt.clf()
wavelengths_to_plot = [365, 430,530,625,730,850,970,1050]
# wavelengths_to_plot = [400]
wavelegth_idx = [np.argmin(np.abs(wavelength - w)) for w in wavelengths_to_plot]
for i in range(len(wavelegth_idx)):
    idx = wavelegth_idx[i]
    plt.plot(aoi, topcon_result['IAM'][idx, :] * 100, label='TOPCon - {} nm'.format(wavelength[idx]),
             color=color_list[i])
    plt.plot(aoi, hjt_result['IAM'][idx,:]*100, label='HJT - {} nm'.format(wavelength[idx]),
             color=color_list[i], linestyle='dashed')

plt.grid('on')
plt.show()
plt.legend()
plt.ylim([95,102])

# AM1.5G IAM curve
am1p5 = metrics.get_AM1p5_spectrum(wavelength=wavelength)
am1p5.keys()

ref_spectrum = am1p5[ 'Global tilt  W*m-2*nm-1']

# Integrate the AM1.5G spectrum with the EQE.
photon_energy = 1 / wavelength
photon_energy = np.transpose(np.atleast_2d(photon_energy))
photocurrent_topcon = np.trapezoid(topcon_result['EQE'] * np.transpose(np.atleast_2d(ref_spectrum)) / photon_energy, x=wavelength, axis=0)
iam_topcon = photocurrent_topcon / photocurrent_topcon[0]

photocurrent_hjt = np.trapezoid(hjt_result['EQE'] * np.transpose(np.atleast_2d(ref_spectrum)) / photon_energy, x=wavelength, axis=0)
iam_hjt = photocurrent_hjt / photocurrent_hjt[0]
plt.figure(2)
plt.clf()
plt.plot(aoi, iam_topcon*100, label='TOPCon')
plt.plot(aoi, iam_hjt*100, label='HJT')
plt.show()
plt.legend()
plt.ylim([95,102])
plt.grid('on')
plt.ylabel('IAM (%)')
plt.xlabel('AOI (degrees)')


# Get spectra
soln = [{'LED Name': 'Intensity (W/m2/nm) - LED0', 'LED Number': 'Intensity (W/m2/nm) - LED0', 'popt0 - Center 1 (nm)': np.float64(367.49365494364207), 'popt1 - Sigma 1 (nm)': np.float64(3.5348772895968366), 'popt2 - Amplitude 1': np.float64(0.0005217004832914629), 'popt3 - Center 2 (nm)': np.float64(371.60651973514126), 'popt4 - Sigma 2 (nm)': np.float64(7.057823341213551), 'popt5 - Amplitude 2': np.float64(0.0001937548305744477)}, {'LED Name': 'Intensity (W/m2/nm) - LED1', 'LED Number': 'Intensity (W/m2/nm) - LED1', 'popt0 - Center 1 (nm)': np.float64(435.2542044560931), 'popt1 - Sigma 1 (nm)': np.float64(12.753367965390103), 'popt2 - Amplitude 1': np.float64(0.000515917547757749), 'popt3 - Center 2 (nm)': np.float64(431.9817686789928), 'popt4 - Sigma 2 (nm)': np.float64(5.146426697095969), 'popt5 - Amplitude 2': np.float64(0.000826344581865185)}, {'LED Name': 'Intensity (W/m2/nm) - LED2', 'LED Number': 'Intensity (W/m2/nm) - LED2', 'popt0 - Center 1 (nm)': np.float64(524.1028834306773), 'popt1 - Sigma 1 (nm)': np.float64(10.447249530498), 'popt2 - Amplitude 1': np.float64(0.000445788351364239), 'popt3 - Center 2 (nm)': np.float64(532.0744513919857), 'popt4 - Sigma 2 (nm)': np.float64(21.887844468050528), 'popt5 - Amplitude 2': np.float64(0.0002961703316569094)}, {'LED Name': 'Intensity (W/m2/nm) - LED3', 'LED Number': 'Intensity (W/m2/nm) - LED3', 'popt0 - Center 1 (nm)': np.float64(630.4390023667712), 'popt1 - Sigma 1 (nm)': np.float64(8.42645268403914), 'popt2 - Amplitude 1': np.float64(0.0031379438338484016), 'popt3 - Center 2 (nm)': np.float64(628.2758657905657), 'popt4 - Sigma 2 (nm)': np.float64(7.0240906287073015), 'popt5 - Amplitude 2': np.float64(-0.0022683335439883817)}, {'LED Name': 'Intensity (W/m2/nm) - LED4', 'LED Number': 'Intensity (W/m2/nm) - LED4', 'popt0 - Center 1 (nm)': np.float64(724.5308897738995), 'popt1 - Sigma 1 (nm)': np.float64(19.416585896395986), 'popt2 - Amplitude 1': np.float64(0.00035939554039373083), 'popt3 - Center 2 (nm)': np.float64(732.2556882548057), 'popt4 - Sigma 2 (nm)': np.float64(8.900418114551474), 'popt5 - Amplitude 2': np.float64(0.000334016907647577)}, {'LED Name': 'Intensity (W/m2/nm) - LED5', 'LED Number': 'Intensity (W/m2/nm) - LED5', 'popt0 - Center 1 (nm)': np.float64(852.4244384032148), 'popt1 - Sigma 1 (nm)': np.float64(7.737953736222358), 'popt2 - Amplitude 1': np.float64(0.0004619668504918907), 'popt3 - Center 2 (nm)': np.float64(840.1213740587766), 'popt4 - Sigma 2 (nm)': np.float64(18.569878004117786), 'popt5 - Amplitude 2': np.float64(0.000257034884445371)}, {'LED Name': 'Intensity (W/m2/nm) - LED6', 'LED Number': 'Intensity (W/m2/nm) - LED6', 'popt0 - Center 1 (nm)': np.float64(942.4250774780453), 'popt1 - Sigma 1 (nm)': np.float64(26.43700259545812), 'popt2 - Amplitude 1': np.float64(9.989583713029071e-05), 'popt3 - Center 2 (nm)': np.float64(968.560807583351), 'popt4 - Sigma 2 (nm)': np.float64(12.949466660102475), 'popt5 - Amplitude 2': np.float64(0.00012826107256602395)}, {'LED Name': 'Intensity (W/m2/nm) - LED7', 'LED Number': 'Intensity (W/m2/nm) - LED7', 'popt0 - Center 1 (nm)': np.float64(1047.1416808213216), 'popt1 - Sigma 1 (nm)': np.float64(14.002178253687086), 'popt2 - Amplitude 1': np.float64(0.00016562393729535382), 'popt3 - Center 2 (nm)': np.float64(1011.24139094762), 'popt4 - Sigma 2 (nm)': np.float64(28.681825714498068), 'popt5 - Amplitude 2': np.float64(8.874733702872873e-05)}]
def gaussian(x, mu, sigma, A):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def sum_of_gaussians(x, *params):
    N = int(len(params) / 3)
    y = np.zeros_like(x).astype(float)
    for i in range(N):
        mu = params[i * 3]
        sigma = params[i * 3 + 1]
        A = params[i * 3 + 2]
        y += gaussian(x, mu, sigma, A)
    return y

def fit_gaussians(x, y, N, p0=None):
    p0 = np.zeros(N * 3)
    for i in range(N):
        p0[i * 3] = x[np.argmax(y)]
        p0[i * 3 + 1] = 10
        p0[i * 3 + 2] = np.max(y)
    popt, pcov = curve_fit(sum_of_gaussians, x, y, p0=p0)
    return popt

N = 2

plt.figure(3,figsize=(3.5,3))
plt.clf()


plt.figure(4,figsize=(3.5,3))
plt.clf()

for sol in soln:

    spectrum = sum_of_gaussians(wavelength,
                                sol['popt0 - Center 1 (nm)'], sol['popt1 - Sigma 1 (nm)'], sol['popt2 - Amplitude 1'],
                                sol['popt3 - Center 2 (nm)'], sol['popt4 - Sigma 2 (nm)'], sol['popt5 - Amplitude 2'])

    plt.figure(3)
    plt.plot(wavelength, spectrum, label=sol['LED Name'],
             color=color_list[soln.index(sol)], linewidth=1,)

    topcon.aoi_series = topcon.aoi_list
    topcon.spectrum_series = np.repeat(np.atleast_2d(spectrum), len(topcon.aoi_series), axis=0).transpose()
    topcon_result = topcon.calculate_series()

    plt.figure(4)
    iam = topcon_result['Photocurrent'] / topcon_result['Photocurrent'][0]
    plt.plot(topcon.aoi_series, iam*100, label='TOPCon ' + sol['LED Name'].split(' - ')[-1],
        color=color_list[soln.index(sol)], linewidth=1,)

    hjt.aoi_series = topcon.aoi_list
    hjt.spectrum_series = np.repeat(np.atleast_2d(spectrum), len(hjt.aoi_series), axis=0).transpose()
    hjt_result = hjt.calculate_series()
    iam = hjt_result['Photocurrent'] / hjt_result['Photocurrent'][0]
    plt.plot(hjt.aoi_series, iam*100, label='HJT ' + sol['LED Name'].split(' - ')[-1],
        color=color_list[soln.index(sol)], linewidth=1, linestyle='dashed')


plt.figure(3)
plt.show()

plt.figure(4)
plt.ylim([95,103])
plt.grid('on')
plt.ylabel('IAM (%)',fontsize=8)
plt.xlabel('AOI (degrees)',fontsize=8)
plt.legend(fontsize=7)
plt.show()
plt.savefig('figures/iam_comparison_with_leds.png',
            dpi=450,
            bbox_inches='tight')