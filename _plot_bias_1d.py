from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

from data import Data
from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

runs = {
        'b8f4e40091ee24e646bb879d225865f6': \
            {
             'fiducial': 'fiducial',
             'mbias/mbias_plus': 'mbias_plus',
             'mbias/mbias_minus': 'mbias_minus',
            },
       }

def make_label (run_hash, bias_info) :
    return f'$\\tt{{ {run_hash[:4]} }}$: {bias_info}'

S8_range = [0.45, 1.05]
S8_fid = Data().get_cosmo('fiducial')[0]

edges = np.linspace(*S8_range, num=51)
centers = 0.5 * (edges[1:] + edges[:-1])
fine_centers = np.linspace(*S8_range, num=500)

fig, ax = plt.subplots(figsize=(5, 5))

for ii, (run_hash, bias_cases) in enumerate(runs.items()) :
    
    all_std = []

    for jj, (bias_case, bias_info) in enumerate(bias_cases.items()) :
        
        fnames = glob(f'{ROOT}/{bias_case}_{run_hash}/bias_data_[0-9]*.dat')
        indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in fnames])
        means, stds = np.concatenate([np.loadtxt(fname, usecols=(1, 2, )) for fname in fnames], axis=0).T
        _, where_unique = np.unique(indices, return_index=True)
        means = means[where_unique]
        stds = stds[where_unique]

        this_std = np.median(stds)
        all_std.append(this_std)

        # h, _  = np.histogram(means, bins=edges)
        # h = h.astype(float) / np.sum(h)
        
        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01*this_std).fit(means.reshape(-1, 1))
        logh = kde.score_samples(fine_centers.reshape(-1, 1))
        logh -= np.max(logh)
        h = np.exp(logh)

        ax.plot(fine_centers, h,
                linestyle=default_linestyles[ii], color=default_colors[jj],
                label=make_label(run_hash, bias_info))

    std = np.median(all_std)
    ax.plot(fine_centers, np.exp( -0.5 * ( (fine_centers-S8_fid) / std )**2 ),
            linestyle=default_linestyles[ii], color=black)

ax.axvline(S8_fid, color=black)
ax.set_xlim(*S8_range)
ax.set_ylim(0, None)
ax.legend(frameon=False)

savefig(fig, 'bias_1d')
