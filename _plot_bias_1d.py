from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from data import Data
from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

runs = {
        'b8f4e40091ee24e646bb879d225865f6': \
            {
             'fiducial': 'fiducial',
            },
       }

def make_label (run_hash, bias_info) :
    return f'$\\tt{{ {run_hash[:4]} }}$: {bias_info}'

S8_range = [0.45, 1.05]
S8_fid = data().get_cosmo('fiducial')[0]

edges = np.linspace(*S8_range, num=51)
centers = 0.5 * (edges[1:] + edges[:-1])

fig, ax = plt.subplots(figsize=(5, 5))

for ii, (run_hash, bias_cases) in enumerate(runs.items()) :

    for jj, (bias_case, bias_info) in enumerate(bias_cases.items()) :
        
        fnames = glob(f'{ROOT}/{bias_case}/bias_data_[0-9]*.dat')
        indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in fnames])
        means, stds = np.concatenate([np.loadtxt(fname, usecols=(1, 2, )) for fname in fnames], axis=0).T
        _, where_unique = np.unique(indices, return_index=True)
        means = means[where_unique]
        stds = stds[where_unique]

        std = np.median(stds) # currently unused

        h, _  = np.histogram(means, bins=edges)
        h = h.astype(float) / len(h)

        ax.plot(centers, h,
                linestyle=default_linestyles[ii], color=default_colors[jj],
                label=make_label(run_hash, bias_info))

ax.axvline(S8_fid, color=black)
ax.set_xlim(*S8_range)
ax.set_ylim(0, None)
ax.legend(frameon=False)

savefig(fig, 'bias_1d')
