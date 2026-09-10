"""
Example processing RGB microscopy image with combined figure and 2 mm scale bars.
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pvarc.image import calculate_thickness, calculate_porosity, calculate_swpr
import os
import glob
import pandas as pd
import tqdm


# ---------- Helper: draw scale bar ----------
def add_scale_bar(ax, length_px, height_px=6, pad_px=20, color='white', edgecolor='black', lw=1.0, loc='lower right'):
    """
    Draws a solid rectangle as a scalebar of given pixel length on the axes.
    loc: 'lower right' | 'lower left' | 'upper right' | 'upper left'
    """
    # Get image size
    h, w = ax.images[0].get_array().shape[:2]
    if loc.startswith('lower'):
        y0 = h - pad_px - height_px
    else:
        y0 = pad_px
    if loc.endswith('right'):
        x0 = w - pad_px - length_px
    else:
        x0 = pad_px

    rect = Rectangle((x0, y0), length_px, height_px,
                     facecolor=color, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(rect)

    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])  # image origin at top-left


# ---------- Input ----------

filename_spec = os.path.join(os.path.expanduser('~'), 'Pictures', '2025_08_08_Nextracker_10287', 'Aging Images','*','*','*.png')

save_dir = os.path.join(os.path.expanduser('~'), 'Pictures', '2025_08_08_Nextracker_10287', 'processed')

filenames = glob.glob(filename_spec)
filenames.sort()

data_accumulator = []

display_scale = 1.25

for filename in tqdm.tqdm(filenames):
    print(f'Processing file: {filename}')

    module = os.path.basename(os.path.dirname(filename))
    location = os.path.basename(filename).replace('.png', '')
    stress_step = os.path.basename(os.path.dirname(os.path.dirname(filename)))


    # ---------- Load image ----------
    im = plt.imread(filename)

    # ---------- Chromaticity + masks ----------
    x = im[:, :, 0] / np.sum(im, axis=2)
    y = im[:, :, 1] / np.sum(im, axis=2)
    mask = im.sum(axis=2) < np.percentile(im.sum(axis=2), 5)

    # ---------- Compute maps ----------
    swpr = calculate_swpr(x, y, porosity_min=0.1, porosity_max=0.45)
    thickness = calculate_thickness(x, y, porosity_min=0.1, porosity_max=0.45)
    porosity = calculate_porosity(x, y, porosity_min=0.1, porosity_max=0.45)

    swpr[mask] = np.nan
    thickness[mask] = np.nan
    porosity[mask] = np.nan

    # ---------- Stats ----------
    print('Mean SWPR: {:.2%}'.format(np.nanmean(swpr)))
    print('Mean Thickness (nm): {:.2f}'.format(np.nanmean(thickness)))
    print('Mean Porosity: {:.2%}'.format(np.nanmean(porosity)))

    # ---------- Scale conversion ----------
    pix_per_mm = 2035 / 8       # your calibration
    pix_per_2mm = pix_per_mm * 2  # 2 mm in pixels

    # ---------- Combined figure ----------
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5), dpi=200, constrained_layout=True)

    # RGB
    ax = axes[0]
    ax.imshow(im * display_scale, origin='upper')
    ax.set_title('{} {} {}'.format(stress_step, module, location))
    ax.set_axis_off()
    # ax.text(0.5, 0.95, '{} {} {}'.format(stress_step, module, location),
    #         transform=ax.transAxes, ha='center', va='top', color='white',
    #         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    add_scale_bar(ax, int(pix_per_2mm))

    # Thickness (nm)
    ax = axes[1]
    t_im = ax.imshow(thickness, origin='upper')
    ax.set_title('Thickness (nm)')
    ax.set_axis_off()
    ax.text(0.5, 0.95, 'Mean Thickness: {:.1f} nm'.format(np.nanmean(thickness)),
            transform=ax.transAxes, ha='center', va='top', color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    plt.colorbar(t_im, ax=ax, fraction=0.046, pad=0.04)

    add_scale_bar(ax, int(pix_per_2mm))

    # Porosity (%)
    ax = axes[2]
    p_im = ax.imshow(porosity * 100.0, origin='upper')
    ax.set_title('Porosity (%)')
    ax.text(0.5, 0.95, 'Mean Porosity: {:.1%}'.format(np.nanmean(porosity)),
            transform=ax.transAxes, ha='center', va='top', color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    ax.set_axis_off()
    plt.colorbar(p_im, ax=ax, fraction=0.046, pad=0.04)
    add_scale_bar(ax, int(pix_per_2mm))

    # SWPR (%)
    ax = axes[3]
    s_im = ax.imshow(swpr * 100.0, vmin=0, vmax=4, origin='upper')
    ax.text(0.5, 0.95, 'Mean SWPR: {:.2%}'.format(np.nanmean(swpr)),
            transform=ax.transAxes, ha='center', va='top', color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    ax.set_title('SWPR (%)')
    ax.set_axis_off()
    cbar = plt.colorbar(s_im, ax=ax, fraction=0.046, pad=0.04)


    add_scale_bar(ax, int(pix_per_2mm))

    # plt.show()
    plt.savefig(os.path.join(save_dir, 'analyzed_{}_{}_{}.jpg'.format(stress_step, module, location)),
                dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    plt.figure(2, figsize=(2,2), dpi=200)
    ax = plt.axes()
    # ax.imshow(im / (np.percentile(im,95)*0.90), origin='upper')
    ax.imshow(im, origin='upper')
    ax.imshow(im * display_scale, origin='upper')
    # ax.set_title('{} {} {}'.format(stress_step, module, location))
    ax.text(0.5, 0.95, '{} {} {}'.format(stress_step, module, location),
            transform=ax.transAxes, ha='center', va='top', color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none',
                      boxstyle='round,pad=0.1'))
    ax.set_axis_off()
    # ax.text(0.5, 0.95, '{} {} {}'.format(stress_step, module, location),
    #         transform=ax.transAxes, ha='center', va='top', color='white',
    #         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    add_scale_bar(ax, int(pix_per_2mm))

    plt.savefig(os.path.join(save_dir, 'rgb_{}_{}_{}_thickness.jpg'.format(stress_step, module, location)),
                dpi=600, bbox_inches='tight',
                pad_inches=0)

    data = {
        'filename': filename,
        'Stress Step': stress_step,
        'Module': module,
        'Location': location,
        'SWPR Mean (%)': np.nanmean(swpr) * 100.0,
        'SWPR Median (%)': np.nanmedian(swpr) * 100.0,
        'Thickness Mean (nm)': np.nanmean(thickness),
        'Thickness Median (nm)': np.nanmedian(thickness),
        'Porosity Mean (%)': np.nanmean(porosity) * 100.0,
        'Porosity Median (%)': np.nanmedian(porosity) * 100.0,
        'SWPR Std (%)': np.nanstd(swpr) * 100.0,
        'Thickness Std (nm)': np.nanstd(thickness),
        'Porosity Std (%)': np.nanstd(porosity) * 100.0,
    }

    data_accumulator.append(data)

    df = pd.DataFrame(data_accumulator)
df.to_csv(os.path.join(save_dir, '00_processed_data.csv'), index=False)



