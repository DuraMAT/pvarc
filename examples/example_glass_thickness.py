"""
Compare EQE and IAM for 2 mm vs 4 mm glass using pvarc.
"""

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

# Wavelength and AOI ranges
wavelength = np.arange(300, 1201, 1)
aoi = np.arange(0, 90, 1)

# Indices
index_glass_coating = refractive_index_porous_silica(wavelength, porosity=0.28)
index_glass = refractive_index_glass(wavelength)
index_encapsulant = refractive_index_eva(wavelength)
index_cell_coating = refractive_index_SiN(wavelength)
index_cell = refractive_index_silicon(wavelength)

# Stack constants
thickness_cell = 300e3  # nm
cell_arc_physical_improvement_factor = 7
thickness_cell_coating = 70   # nm
thickness_encapsulant = 0.45e-3 * 1e9  # nm
thickness_glass_coating = 140  # nm
index_air = 1.0003

# Glass thicknesses to compare
glass_thicknesses = {
    "2 mm": 2.0e-3 * 1e9,   # nm
    "4 mm": 4.0e-3 * 1e9
}

results = {}

# Run simulations
for label, thickness_glass in glass_thicknesses.items():
    absorbance = np.zeros((len(aoi), len(wavelength)))
    for k, angle in enumerate(aoi):
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
            aoi=angle,
            polarization="mixed",
            index_air=index_air,
        )
        absorbance[k, :] = ret["EQE"]

    # IAM relative to AOI = 0
    iam = absorbance / absorbance[0, :]
    results[label] = {"EQE": absorbance, "IAM": iam, "ret": ret}

# --- Plots ---
wavelength_to_plot = [365, 430, 530, 625, 730, 850, 970, 1050]
idx_to_plot = [np.argmin(np.abs(wa - wavelength)) for wa in wavelength_to_plot]
color_list = ['#9400D3','#4B0082','#0000FF','#00FF00','#CCCC00','#FF7F00','#FF0000','#A52A2A']

# IAM comparison
plt.figure(figsize=(6,4))
for label, res in results.items():
    for k, idx in enumerate(idx_to_plot):
        plt.plot(aoi, 100*res["IAM"][:, idx],
                 label=f"{label} {wavelength[idx]} nm",
                 color=color_list[k], linestyle="--" if "4 mm" in label else "-")
plt.xlabel("AOI (degrees)")
plt.ylabel("IAM (%)")
plt.legend(fontsize=7, ncol=2)
plt.grid(True)
plt.ylim([90, 102])
plt.xlim([0, 90])
plt.title("IAM vs. AOI for 2 mm vs 4 mm glass")
plt.tight_layout()
plt.savefig("iam_comparison.png", dpi=300)
plt.show()

# EQE comparison (AOI = 0)
plt.figure(figsize=(6,4))
for label, res in results.items():
    plt.plot(wavelength, res["EQE"][0,:], label=f"{label}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("EQE (at AOI=0)")
plt.legend()
plt.grid(True)
plt.xlim([300, 1200])
plt.ylim([0, 1])
plt.title("EQE vs Wavelength for 2 mm vs 4 mm glass")
plt.tight_layout()
plt.savefig("eqe_comparison.png", dpi=300)
plt.show()
