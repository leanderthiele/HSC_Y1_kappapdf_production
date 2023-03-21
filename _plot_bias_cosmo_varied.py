from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import Data
from settings import CHAIN_ROOT
from _plot_style import *

run_hash = '9d56790a0f55a6885899ec32284b91bd'

fnames = glob(f'{CHAIN_ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
indices = np.concatenate([np.loadtxt(fname, usecols=(0, ), dtype=int) for fname in fnames])
mean, std = np.concatenate([np.loadtxt(fname, usecols=(6, 5)) for fname in fnames], axis=0).T
_, select = np.unique(indices, return_index=True)
indices = indices[select]
mean = mean[select]
std = std[select]

data = Data()
cosmo_indices = np.array([idx//data.get_nseeds('cosmo_varied') for idx in indices], dtype=int)
theta = data.get_cosmo('cosmo_varied')

uniq_cosmo_indices = np.unique(cosmo_indices)
avg_bias = []
N_points = []
for uniq_cosmo_idx in uniq_cosmo_indices :
    select = (uniq_cosmo_idx == cosmo_indices)
    N_points.append(np.count_nonzero(select))
    avg_bias.append( np.median( (mean[select] - theta[uniq_cosmo_idx][0]) / std[select] ) )
avg_bias = np.array(avg_bias)

fig, ax = plt.subplots(figsize=(3, 4))
vmax = 1
im = ax.scatter(*theta[uniq_cosmo_indices].T, c=avg_bias, vmin=-vmax, vmax=vmax,
                cmap='seismic', edgecolors=black)
for idx, b, n in zip(uniq_cosmo_indices, avg_bias, N_points) :
    ax.text(theta[idx][0], theta[idx][1]+3e-3,
            f'{b:.2f}',
            va='bottom', ha='center',
            fontsize='x-small', transform=ax.transData)

divider = make_axes_locatable(ax)
cax = divider.append_axes('top', size='5%', pad=0.6)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('$\Delta S_8 / \sigma(S_8)$')

ax.set_xlabel('$S_8$')
ax.set_ylabel('$\Omega_m$')
ax.set_yticks([0.2, 0.3, 0.4, ])

savefig(fig, 'bias_cosmo_varied')
