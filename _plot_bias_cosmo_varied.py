from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import Data
from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

run_hash = '5ae39f509acb63122ff1b8b9f2baa589'

fnames = glob(f'{ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
indices = np.concatenate([np.loadtxt(fname, usecols=(0, ), dtype=int) for fname in fnames])
mean, std = np.concatenate([np.loadtxt(fname, usecols=(4, 5)) for fname in fnames], axis=0).T

data = Data()
cosmo_indices = np.array([idx//data.get_nseeds('cosmo_varied') for idx in indices], dtype=int)
theta = data.get_cosmo('cosmo_varied')

uniq_cosmo_indices = np.unique(cosmo_indices)
avg_bias = []
N_points = []
for uniq_cosmo_idx in uniq_cosmo_indices :
    select = (uniq_cosmo_idx == cosmo_indices)
    N_points.append(np.count_nonzero(select))
    avg_mean = np.median( (mean[select] - theta[uniq_cosmo_idx]) / std[select] )
avg_bias = np.array(avg_bias)

fig, ax = plt.subplots(figsize=(5, 5))
vmax = np.max(np.fabs(avg_bias))
im = ax.scatter(*theta[uniq_cosmo_indices].T, c=avg_bias, vmin=-2, vmax=2, cmap='seismic')
for idx, b, n in zip(uniq_cosmo_indices, avg_bias, N_points) :
    ax.text(*theta[idx], f'{b:.2f} {n:d}', va='bottom', ha='center',
            fontsize='xx-small', transform=ax.transData)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('$\Delta S_8 / \sigma(S_8)$')

ax.set_xlabel('$S_8$')
ax.set_ylabel('$Omega_m$')
