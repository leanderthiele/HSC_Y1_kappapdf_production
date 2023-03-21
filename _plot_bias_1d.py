from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

from data import Data
from settings import CHAIN_ROOT
from _plot_style import *

runs = {
        '9d56790a0f55a6885899ec32284b91bd': \
             {
              'fiducial': 'fiducial',
              'mbias/mbias_plus': '$\Delta m = +1\,\%$',
              'mbias/mbias_minus': '$\Delta m = -1\,\%$',
              'fiducial-baryon': 'baryons',
              'photoz/frankenz': '${\\tt frankenz}$',
              'photoz/mizuki': '${\\tt mizuki}$',
              'fiducial-IA/032_simple': '$A_{\sf IA} = -0.32$',
              'fiducial-IA/118_simple': '$A_{\sf IA} = 1.18$',
             },
       }

def make_label (run_hash, bias_info) :
    return bias_info

S8_range = [0.50, 1.00]
S8_fid = Data().get_cosmo('fiducial')[0]

edges = np.linspace(*S8_range, num=51)
centers = 0.5 * (edges[1:] + edges[:-1])
fine_centers = np.linspace(*S8_range, num=500)

fig, ax = plt.subplots(figsize=(5, 5))
fig_bars, ax_bars = plt.subplots(figsize=(2, 4))

# this is just some running index
ycoord = 0
fid_x = None

for ii, (run_hash, bias_cases) in enumerate(runs.items()) :
    
    all_std = []
    if len(bias_cases) == 0 :
        continue

    for jj, (bias_case, bias_info) in enumerate(bias_cases.items()) :
        
        pattern = f'{CHAIN_ROOT}/{bias_case}_{run_hash}/bias_data_[0-9]*.dat'
        print(pattern)
        fnames = glob(pattern)
        indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in fnames])
        means, stds = np.concatenate([np.loadtxt(fname, usecols=(1, 2, )) for fname in fnames], axis=0).T
        _, where_unique = np.unique(indices, return_index=True)
        means = means[where_unique]
        stds = stds[where_unique]

        this_mean = np.median(means)
        this_std = np.median(stds)
        all_std.append(this_std)

        print(f'delta(S8)/sigma = {(np.median(means)-S8_fid)/this_std:.2f} [{run_hash[:4]} {bias_info}]')
        
        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.05).fit(means.reshape(-1, 1))
        logh = kde.score_samples(fine_centers.reshape(-1, 1))
        logh -= np.max(logh)
        h = np.exp(logh)

        label = make_label(run_hash, bias_info)

        ax.plot(fine_centers, h,
                linestyle=default_linestyles[ii], color=default_colors[jj],
                label=label)

        ax_bars.errorbar([this_mean, ], [ycoord,], xerr=this_std,
                         label=label, marker='^', color=black)
        if ycoord == 0 :
            ax_bars.axvline(this_mean, color='grey', linestyle='dashed')
            fid_x = this_mean
        ax_bars.text(fid_x+2e-3, ycoord+1e-2, label, ha='left', va='bottom', transform=ax_bars.transData)
        ycoord -= 1

    std = np.median(all_std)
    print(f'std = {std:.3f} [{run_hash[:4]}]')
    ax.plot(fine_centers, np.exp( -0.5 * ( (fine_centers-S8_fid) / std )**2 ),
            linestyle=default_linestyles[ii], color=black)

ax.set_xlabel('$S_8$')
ax.axvline(S8_fid, color=black)
ax.set_xlim(*S8_range)
ax.set_ylim(0, None)
ax.legend(frameon=False)

for s in ['top', 'left', 'right', ] :
    ax_bars.spines[s].set_visible(False)
ax_bars.set_xlabel('$S_8$')
ax_bars.set_yticks([])

savefig(fig, 'bias_1d')
savefig(fig_bars, 'bias_bars')
