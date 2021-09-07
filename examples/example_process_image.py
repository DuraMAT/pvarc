import numpy as np
import pandas as pd
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvarc.image import build_rgb_to_thickness_porosity_interpolator_data, \
    calculate_thickness, calculate_porosity, calculate_swpr, calculate_rgb, \
    get_thickness_porosity_interpolator_data
from scipy.optimize import minimize

filename = '../data/08_MisControlF2UR-1M_crop.tif'
# filename = '../data/02_CanControlF2UR-1L.tif'

# Load image
im = plt.imread(filename)

# Inspect image
plt.figure(1)
plt.clf()
plt.imshow(im / im.max())
plt.show()

# Get data
df = get_thickness_porosity_interpolator_data()
thickness_max = 200
porosity_set = 0.3
dfp = df[np.logical_and(df['porosity'] == porosity_set, df['thickness']<thickness_max)]
thickness_smooth = np.array(dfp['thickness'])
swpr_smooth = np.array(dfp['swpr'])
rgbt = np.array(dfp[['R', 'G', 'B']])

# Normalize image
imn = im / np.max(im)

mask = imn.sum(axis=2) < np.percentile(imn.sum(axis=2),20)

# Repeated versions
imr = np.repeat(imn[:, :, np.newaxis, :].astype('float'), len(thickness_smooth),
                axis=2)
# rgbtr = rgbt[np.newaxis, np.newaxis, :, :]

rgbtr = np.repeat(np.repeat(rgbt[np.newaxis, np.newaxis, :, :],imr.shape[0],axis=0),imr.shape[1],axis=1)


def residual(norm):
    thickness_idx = np.argmin(np.mean((rgbtr - norm * imr) ** 2, axis=3),
                              axis=2)

    color_loss = np.mean((rgbt[thickness_idx] - norm * imn) ** 2)

    print('Norm: {}, Color loss: {}'.format(norm, color_loss))
    return color_loss


optimize = False
if optimize:
    # Find best normalization.
    ret = minimize(residual, x0=1.1,
                   options={'disp': True})

    norm_best = float(ret['x'])
else:
    norm_best = 0.7

start_time = time.time()
thickness_idx = np.argmin(np.mean((rgbtr - norm_best * imr) ** 2, axis=3),
                          axis=2)
print('Elapsed time: {} s'.format(time.time() - start_time))
thickness = thickness_smooth[thickness_idx]
thickness[mask]= np.nan
swpr = swpr_smooth[thickness_idx]
swpr[mask]= np.nan

plt.figure(11)
plt.clf()
ax = plt.axes()
plt.pcolor(thickness,cmap='jet',vmin=0,vmax=thickness_max)
cbar = plt.colorbar()
cbar.set_label('Thickness (nm)')
plt.gca().invert_yaxis()
plt.show()

# plt.savefig('{}_thickness_{:.3f}.png'.format(filename_list[0],norm_best).replace('/','_'),
#             bbox_inches='tight',
#             dpi=300)


plt.figure(12)
plt.clf()
ax = plt.axes()
plt.pcolor(swpr*100)
cbar = plt.colorbar()
cbar.set_label('SWPR (%)')
plt.gca().invert_yaxis()
plt.show()


# Calculate thickness using chromaticity method
x = imn[:, :, 0] / np.sum(imn, axis=2)
y = imn[:, :, 1] / np.sum(imn, axis=2)
thickness_c = calculate_thickness(x, y, porosity_min=0.1, porosity_max=0.45)
thickness_c[mask]= np.nan


plt.figure(21)
plt.clf()
ax = plt.axes()
plt.pcolor(thickness_c,cmap='jet',vmin=0,vmax=thickness_max)
cbar = plt.colorbar()
cbar.set_label('Thickness (nm)')
plt.gca().invert_yaxis()
plt.show()
#


# Calculate SWPR using chromaticity method
swpr_c = calculate_swpr(x, y, porosity_min=0.1, porosity_max=0.45)
swpr_c[mask]= np.nan

print('Mean SWPR: {:.2%}'.format(np.nanmean(swpr_c)))
plt.figure(22)
plt.clf()
ax = plt.axes()
# plt.contourf(thickness_c, cmap='jet', levels=20, vmin=0, vmax=thickness_max)
plt.pcolor(swpr_c*100)
cbar = plt.colorbar()
cbar.set_label('SWPR (%)')
plt.gca().invert_yaxis()
plt.show()

df = get_thickness_porosity_interpolator_data()

calculate_swpr(1/3,1/3)