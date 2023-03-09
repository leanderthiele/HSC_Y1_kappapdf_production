from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# have placeholder which we can later use for labels
runs = {
        # not perfect but ok
        'befab23d6ee10fe971a5ad7118957c9c': 'baseline PDF only',
        # clearly overconfident
        # '3e14c0d1a34c1aa19ab78949396014de': 'cov_mode=gpr_scale',
        # clearly overconfident
        # '45b6a5961cb86a0e7690fc6919c9bf8e': 'cov_mode=fixed',
        # somewhat overconfident
        'd9d31391c8a306a481a4a26ce07969d2': 'baseline PDF+PS',
        # better tha d9d3 but still overconfident
        'b8f4e40091ee24e646bb879d225865f6': 'PDF+PS, PS cov_mode=fixed',
        #
        '68c282161ba83a2267303b9ea1500119': 'PS only, cov_mode=fixed',
       }

# can play with the binning here
Nbins_qq = 50
Nbins_ra = 20
edges_qq, edges_ra = [np.linspace(0, 1, num=Nbins+1) for Nbins in [Nbins_qq, Nbins_ra, ]]

fig_qq, ax_qq = plt.subplots(figsize=(5, 5))
fig_ra, ax_ra = plt.subplots(figsize=(5, 5))

for run_hash, run_info in runs.items() :
    
    fnames = glob(f'{ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
    # figure out if this is one of the old runs where we didn't have the index information
    with open(fnames[0], 'r') as f :
        header = f.readline().strip()
    col_offset = 1 if 'index' in header else 0
    # do not filter for unique runs here as it improves the faithfulness of our prior
    # sampling if the prior is non-trivial
    # idx = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in fnames])
    oma, ranks = np.concatenate([
                                 np.loadtxt(fname, usecols=(col_offset+0, col_offset+1))
                                 for fname in fnames
                                ], axis=0).T

    # filter out failures (shouldn't be many)
    oma = oma[oma>0]
    ranks = ranks[ranks>0]

    # coverage = np.array([np.count_nonzero(oma<e) for e in edges_qq]) / len(oma)

    label = f'$\\tt{{ {run_hash[:4]} }}$: {run_info}'
    ax_qq.hist(oma, bins=edges_qq, histtype='step', cumulative=True, density=True,
               label=label)
    ax_ra.hist(ranks, bins=edges_ra, histtype='step', density=True,
               label=label)

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
