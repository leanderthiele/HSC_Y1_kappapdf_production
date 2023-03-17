from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

from data import Data
from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

runs = {
        # PDF + PS baseline, relatively ok, largest shift is mbias_minus which is 0.3 sigma
        'b8f4e40091ee24e646bb879d225865f6': \
             {
              'fiducial': 'fiducial',
              'mbias/mbias_plus': '$\Delta m$\n $+1\%$',
              'mbias/mbias_minus': '$\Delta m$\n $-1\%$',
              'fiducial-baryon': 'baryons',
              'photoz/frankenz': 'photo-$z$\n ${\\tt frankenz}$',
              'photoz/mizuki': 'photo-$z$\n ${\\tt mizuki}$',
             },

        # PDF baseline, all fine (shifts within 0.1 sigma)
        # interesting: width of the posterior depends on mbias:
        #      larger mbias -> tighter posterior. Can we explain this?
        #'befab23d6ee10fe971a5ad7118957c9c': \
        #    {
        #     'fiducial': 'fiducial',
        #     'mbias/mbias_plus': 'mbias_plus',
        #     'mbias/mbias_minus': 'mbias_minus',
        #     'fiducial-baryon': 'baryon',
        #    },

        # PS baseline
        #'68c282161ba83a2267303b9ea1500119': \
        #    {
        #     'fiducial': 'fiducial',
        #    },

        # PDF one more low bin
        # still fine, but improvement in error bar is very small (0.102 vs 0.103)
        # so no need to include this bin
        # '9fe279192f2aa13b590e3367731e7a60': {  'fiducial-baryon': 'baryon', },

        # PDF + PS, ps high_cut=5 instead of 6
        # starts to shift visibly, 0.3 in units of sigma, and only small improvement in constraint
        # '496e0da40dc63eb1faa88522765de834': { 'fiducial-baryon': 'baryon', },

        # PDF + PS, with the point where compression derivatives are evaluated shifted
        # by +0.05 in both directions
        # '5ae39f509acb63122ff1b8b9f2baa589': { 'fiducial': 'fiducial' },

        # same but shift = -0.05
        # '6e8363dcd1644fdc55c7dee18e98cdd5': { 'fiducial': 'fiducial' },

        # use only half the cosmo varied augmentations when estimating mean emulator
        # 'e23a7da97c82e388c290089405629e2e': { 'fiducial': 'fiducial' },

        # rbf_length_scale = 3
        # 'b1820713b3b511d2c9e67c482b07e1b2': { 'fiducial': 'fiducial' },
       }

def make_label (run_hash, bias_info) :
#    return f'$\\tt{{ {run_hash[:4]} }}$: {bias_info}'
    return bias_info

S8_range = [0.50, 1.00]
S8_fid = Data().get_cosmo('fiducial')[0]

edges = np.linspace(*S8_range, num=51)
centers = 0.5 * (edges[1:] + edges[:-1])
fine_centers = np.linspace(*S8_range, num=500)

fig, ax = plt.subplots(figsize=(5, 5))
fig_bars, ax_bars = plt.subplots(figsize=(2, 5))

# this is just some running index
ycoord = 0
fid_x = None

for ii, (run_hash, bias_cases) in enumerate(runs.items()) :
    
    all_std = []
    if len(bias_cases) == 0 :
        continue

    for jj, (bias_case, bias_info) in enumerate(bias_cases.items()) :
        
        fnames = glob(f'{ROOT}/{bias_case}_{run_hash}/bias_data_[0-9]*.dat')
        indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in fnames])
        means, stds = np.concatenate([np.loadtxt(fname, usecols=(1, 2, )) for fname in fnames], axis=0).T
        _, where_unique = np.unique(indices, return_index=True)
        means = means[where_unique]
        stds = stds[where_unique]

        this_mean = np.median(means)
        this_std = np.median(stds)
        all_std.append(this_std)

        print(f'delta(S8)/sigma = {(np.median(means)-S8_fid)/this_std:.2f} [{run_hash[:4]} {bias_info}]')

        # h, _  = np.histogram(means, bins=edges)
        # h = h.astype(float) / np.sum(h)
        
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
        #ax_bars.text(0.9, ycoord, label, ha='left', va='center', transform=ax_bars.transData)
        ax_bars.text(fid_x+1e-3, ycoord, label, ha='left', va='bottom')
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
