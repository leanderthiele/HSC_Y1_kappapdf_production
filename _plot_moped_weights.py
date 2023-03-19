import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from compressed_data import CompressedData
from _plot_style import *
from settings import S

kappa_edges = np.linspace(-4, 4, num=20)
kappa_centers = 0.5*(kappa_edges[1:]+kappa_edges[:-1])
def expand_vec (x) :
    assert S['pdf']['rebin'] == 1
    x = np.concatenate([np.full(S['pdf']['low_cut'], float('nan')), 
                        x,
                        np.full(S['pdf']['high_cut'], float('nan'))])
    x = np.insert(x, S['pdf']['delete'], float('nan'))
    return x

d = CompressedData()

fid_data = None
cv_data = None
matrix = None
for p in d.parts :
    if p[1] is not None :
        assert matrix is None
        matrix = p[1]
        fid_data = p[0].get_datavec('fiducial')
        cv_data = p[0].get_datavec('cosmo_varied')[29]

avg_fid_data = np.mean(fid_data, axis=0)
avg_cv_data = np.mean(cv_data, axis=0)

# pick what corresponds to S8
weights = matrix[0]
datavec = avg_cv_data - avg_fid_data

# Data vector ordering is: [zs, smoothing, bin]
# we reshape the weights as such
new_shape = (len(S['pdf']['zs']), len(S['pdf']['smooth']), -1)
weights = weights.reshape(*new_shape)
datavec = datavec.reshape(*new_shape)
effective_contribs = weights * datavec

fig, ax = plt.subplots(ncols=2, figsize=(5,3))

ax_mat = ax[0]
ax_bins = ax[1]

mat_data = np.sum(effective_contribs, axis=-1)
vmax = np.max(np.fabs(mat_data))
mat_data /= vmax

bins_data = np.sum(effective_contribs.reshape(-1, effective_contribs.shape[-1]), axis=0)
vmax = np.max(np.fabs(bins_data))
bins_data /= vmax
bins_data = expand_vec(bins_data)

im = ax_mat.imshow(mat_data, cmap='seismic', vmin=-1, vmax=1)
divider = make_axes_locatable(ax_mat)
cax = divider.append_axes('top', size='5%', pad=0.6)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('contribution')
cbar.set_ticks([-1, 0, 1])

# smoothing scales
smoothing_scales = [1, 2, 5, 7, 8, 10, 15, 25, ]
xticklabels = [f'$\\theta_s = {smoothing_scales[ii]}\,{{\sf arcmin}}$' for ii in S['pdf']['smooth']]
ax_mat.set_xticks(np.arange(len(xticklabels)))
ax_mat.set_xticklabels(xticklabels)
plt.setp(ax_mat.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

# source redshifts
zs_ranges = [(0.3, 1.5), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5)]
yticklabels = [f'${zs_ranges[ii][0]} < z_s < {zs_ranges[ii][1]}$' for ii in S['pdf']['zs']]
ax_mat.set_yticks(np.arange(len(yticklabels)))
ax_mat.set_yticklabels(yticklabels)

l1 = ax_bins.plot(kappa_centers, bins_data, linestyle='none', marker='o', label='contribution')
y = expand_vec(np.mean(avg_fid_data.reshape(-1, effective_contribs.shape[-1]), axis=0))
y -= np.nanmin(y)
y /= np.nanmax(y)
l2 = ax_bins.plot(kappa_centers, y, linestyle='none', marker='x', label='data vector (shifted & scaled)')
ax_bins.set_xlabel('$\kappa / \sigma(\kappa)$')
ax_bins.axhline(0, color='grey', linestyle='dashed')
# ax_bins.legend(frameon=False)
ax_bins.text(-1, 0.8, 'data\nvector', color=plt.getp(l2[0], 'color'), va='top', ha='left',
             transform=ax_bins.transData)
ax_bins.text(-1, -0.2, 'contribution', color=plt.getp(l1[0], 'color'), va='center', ha='left',
             transform=ax_bins.transData)
ax_bins.set_yticks([0, 1])

savefig(fig, 'moped_weights')
