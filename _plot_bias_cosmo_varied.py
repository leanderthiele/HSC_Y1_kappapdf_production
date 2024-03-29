from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import Data
from settings import CHAIN_ROOT
from _plot_style import *

# this is 9d56 but with subsample=25 turned on
run_hash = 'a98c33f02a1d0e6efa58e589da36f5c5'

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

fig, ax = plt.subplots(figsize=(5, 5))
vmax = 1
im = ax.scatter(*theta[uniq_cosmo_indices].T, c=avg_bias, vmin=-vmax, vmax=vmax,
                cmap='coolwarm', edgecolors=black)
for idx, b, n in zip(uniq_cosmo_indices, avg_bias, N_points) :
    ax.text(theta[idx][0], theta[idx][1]+3e-3,
            f'{b:.2f}',
            va='bottom', ha='center',
            fontsize='x-small', transform=ax.transData)

divider = make_axes_locatable(ax)
if False :
    cax = divider.append_axes('top', size='5%', pad=0.6)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
else :
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('$\Delta S_8 / \sigma(S_8)$')
cbar.set_ticks([-1, 0, 1])

ax.set_xlabel('$S_8$')
ax.set_ylabel('$\Omega_m$')
ax.set_yticks([0.2, 0.3, 0.4, ])
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

savefig(fig, 'bias_cosmo_varied')
