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
        pdf_data = p[0].get_datavec('fiducial')
        cv_data = p[0].get_datavec('cosmo_varied')[37]

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

fig, ax = plt.subplots(nrows=2, figsize=(5,5))

ax_mat = ax[0]
ax_bins = ax[1]

mat_data = np.sum(effective_contribs, axis=-1)
vmax = np.max(np.fabs(mat_data))
mat_data /= vmax

bins_data = np.sum(effective_contribs.reshape(-1, effective_contribs.shape[-1]), axis=0)
vmax = np.max(np.fabs(bins_data))
bins_data /= vmax
bins_data = expand_vec(bins_data)

im = ax_mat.matshow(mat_data, cmap='seismic', vmin=-1, vmax=1)
divider = make_axes_locatable(ax_mat)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('contribution')

ax_bins.plot(kappa_centers, bins_data, linestyle='none', marker='o')

savefig(fig, 'moped_weights')
