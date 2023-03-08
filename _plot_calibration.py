from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# have placeholder which we can later use for labels
runs = {
        'befab23d6ee10fe971a5ad7118957c9c': 'baseline PDF only',
        '3e14c0d1a34c1aa19ab78949396014de': 'cov_mode=gpr_scale',
       }

# can play with the binning here
Nbins_qq = 50
Nbins_ra = 20
edges_qq, edges_ra = [np.linspace(0, 1, num=Nbins+1) for Nbins in [Nbins_qq, Nbins_ra, ]]

fig_qq, ax_qq = plt.subplots(figsize=(5, 5))
fig_ra, ax_ra = plt.subplots(figsize=(5, 5))

for run_hash, run_info in runs.items() :
    
    fnames = glob(f'{ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
    oma = np.concatenate([np.loadtxt(fname, usecols=(0,)) for fname in fnames])
    ranks = np.concatenate([np.loadtxt(fname, usecols=(1,)) for fname in fnames])

    # filter out failures (shouldn't be many)
    oma = oma[oma>0]
    ranks = ranks[ranks>0]

    coverage = np.array([np.count_nonzero(oma<e) for e in edges_qq]) / len(oma)

    label = f'$\\tt{{ {run_hash[:4]} }}$: {run_info}'
    ax_qq.plot(edges_qq, coverage, label=label)
    ax_ra.hist(ranks, bins=edges_ra, histtype='step', label=label)

ax_qq.axline((0, 0), slope=1, color='grey', linestyle='dashed')
ax_qq.set_xlim(0, 1)
ax_qq.set_ylim(0, 1)

ax_ra.set_xlim(0, 1)
ax_ra.set_ylim(0, None)

ax_qq.legend(loc='center right', frameon=False)
ax_qq.set_xlabel('confidence level')
ax_qq.set_ylabel('empirical coverage')
ax_qq.text(0.05, 0.95, 'underconfident', va='top', ha='left', transform=ax_qq.transAxes) 
ax_qq.text(0.95, 0.05, 'overconfident', va='bottom', ha='right', transform=ax_qq.transAxes)

ax_ra.legend(loc='lower center', frameon=False)
ax_ra.set_xlabel('fractional position in chain')
ax_ra.set_ylabel('number of chains')

savefig(fig_qq, 'calibration_qq')
savefig(fig_ra, 'calibration_ranks')
