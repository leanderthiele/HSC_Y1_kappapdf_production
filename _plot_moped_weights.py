import numpy as np
from matplotlib import pyplot as plt

from compressed_data import CompressedData
from _plot_style import *
from settings import S

kappa_edges = np.linspace(-4, 4, num=20)
kappa_centers = 0.5*(kappa_edges[1:]+kappa_edges[:-1])
def expand_vec (x) :

d = CompressedData()

pdf_data = None
matrix = None
for p in d.parts :
    if p[1] is not None :
        assert matrix is None
        matrix = p[1]
        pdf_data = p[0].get_datavec('fiducial')

# pick what corresponds to S8
weights = matrix[0]
datavec = np.mean(pdf_data, axis=0)

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

ax_mat.matshow(mat_data, cmap='seismic', vmin=-1, vmax=1)
ax_bins.plot(bins_data)

savefig(fig, 'moped_weights')
