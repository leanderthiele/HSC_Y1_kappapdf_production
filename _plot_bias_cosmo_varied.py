from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import Data
from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# this is with fiducial settings
# run_hash = 'b8f4e40091ee24e646bb879d225865f6'

# this is with rbf_length_scale = 3
# seems like biases got a bit smaller compared to b8f4
# run_hash = 'b1820713b3b511d2c9e67c482b07e1b2'

# this is with rbf_length_scale = 2
# run_hash = '5b98a5448caccaeb9a18bdead9151b4a'

# this is the test proposed by Jia: train only on 25 cosmo varied augments
# and test on the other 25
run_hash = 'e23a7da97c82e388c290089405629e2e'

fnames = glob(f'{ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
indices = np.concatenate([np.loadtxt(fname, usecols=(0, ), dtype=int) for fname in fnames])
mean, std = np.concatenate([np.loadtxt(fname, usecols=(4, 5)) for fname in fnames], axis=0).T
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
vmax = 1 # np.max(np.fabs(avg_bias))
im = ax.scatter(*theta[uniq_cosmo_indices].T, c=avg_bias, vmin=-vmax, vmax=vmax, cmap='seismic')
for idx, b, n in zip(uniq_cosmo_indices, avg_bias, N_points) :
    ax.text(*theta[idx], f'{b:.2f} {n:d}', va='bottom', ha='center',
            fontsize='xx-small', transform=ax.transData)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('$\Delta S_8 / \sigma(S_8)$')

ax.set_xlabel('$S_8$')
ax.set_ylabel('$\Omega_m$')
ax.set_title(f'{run_hash[:4]}')

savefig(fig, 'bias_cosmo_varied')
