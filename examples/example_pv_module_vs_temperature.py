import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pvarc import pv_stack_absorbance
from pvarc.materials import (
    refractive_index_porous_silica,
    refractive_index_glass,
    refractive_index_silicon,
    refractive_index_eva,
    refractive_index_SiN
)
from pvarc.metrics import get_AM1p5_spectrum

from pvlib.pvsystem import i_from_v, singlediode

# ---------------------------------------------------------------------
# Solar spectrum (AM1.5G, global tilt)
# ---------------------------------------------------------------------
am1p5 = get_AM1p5_spectrum()
wavelength_sun = am1p5["wavelength nm"].values
spectral_irradiance = am1p5["Global tilt  W*m-2*nm-1"].values  # W/m2/nm

# Convert spectral irradiance to photon flux [photons/m2/s/nm]
h = 6.62607015e-34  # J*s
c = 2.99792458e8    # m/s
q = 1.602176634e-19 # C
E_photon = (h * c) / (wavelength_sun * 1e-9)  # J
photon_flux = spectral_irradiance / E_photon  # photons / m2 / s / nm

# ---------------------------------------------------------------------
# Wavelength axis for simulation [nm]
# ---------------------------------------------------------------------
wavelength = np.arange(300, 1201, 1)

# Interpolate photon flux to simulation grid
photon_flux_interp = np.interp(wavelength, wavelength_sun, photon_flux)

# ---------------------------------------------------------------------
# Temperatures to compare
# ---------------------------------------------------------------------
temperatures_C = [15, 25, 50, 75]

# Indices not strongly temperature dependent
index_glass_coating = refractive_index_porous_silica(wavelength, porosity=0.28)
index_glass = refractive_index_glass(wavelength)
index_encapsulant = refractive_index_eva(wavelength)
index_cell_coating = refractive_index_SiN(wavelength)

# Geometry
thickness_cell = 300e3          # nm
thickness_cell_coating = 70     # nm
thickness_encapsulant = 0.45e-3 * 1e9  # nm
thickness_glass = 2e-3 * 1e9    # nm
thickness_glass_coating = 140   # nm
cell_arc_physical_improvement_factor = 7

# ---------------------------------------------------------------------
# EQE curves and Jsc
# ---------------------------------------------------------------------
Jsc_list = []

plt.figure(figsize=(6,4))
plt.clf()

for T_C in temperatures_C:
    index_cell = refractive_index_silicon(wavelength, temperature=T_C)

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
        aoi=0,
        polarization='mixed',
        index_air=1.0003
    )

    EQE = ret['EQE']  # dimensionless
    plt.plot(wavelength, EQE, label=f'{T_C} °C')

    # Calculate Jsc [A/m2] = q * ∫ EQE(λ) * photon_flux(λ) dλ
    Jsc = q * np.trapz(EQE * photon_flux_interp, wavelength * 1e-9)  # integrate over meters
    Jsc_list.append(Jsc / 10)  # A/cm2

# Overlay AM1.5G spectrum (scaled for visualization)
irradiance_norm = spectral_irradiance / np.max(spectral_irradiance)
plt.plot(wavelength_sun, irradiance_norm, 'k--', label='AM1.5G (normalized)')

plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE (a.u.) / Spectrum (norm.)')
plt.title('EQE vs Temperature with AM1.5G')
plt.grid(True)
plt.xlim([300,1200])
plt.ylim([0,1.1])
plt.legend()
plt.tight_layout()
plt.savefig('EQE_AM1p5_vs_Temperature.png', dpi=300)
plt.show()

# ---------------------------------------------------------------------
# Plot Jsc vs Temperature
# ---------------------------------------------------------------------
plt.figure(figsize=(5,3.5))
plt.plot(temperatures_C, Jsc_list, 'o-')
plt.xlabel('Temperature (°C)')
plt.ylabel(r'$J_{sc}$ (A/cm$^2$)')
plt.title('Short-circuit current vs Temperature')
plt.grid(True)
plt.tight_layout()
plt.savefig('Jsc_vs_Temperature.png', dpi=300)
plt.show()
