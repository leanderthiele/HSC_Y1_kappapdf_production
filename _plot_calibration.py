from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from _plot_style import *

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# can play with the binning here
Nbins_qq = 20
Nbins_ra = 20
edges_qq, edges_ra = [np.linspace(0, 1, num=Nbins+1) for Nbins in [Nbins_qq, Nbins_ra, ]]

def PlotCalibration (runs, make_label, qq_legend_kwargs=None, ra_legend_kwargs=None) :
    fig_qq, ax_qq = plt.subplots(figsize=(5, 3))
    fig_ra, ax_ra = plt.subplots(figsize=(5, 3))

    Nsamples = 1500

    for run_hash, run_info in runs.items() :
        
        fnames = glob(f'{ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
        # figure out if this is one of the old runs where we didn't have the index information
        with open(fnames[0], 'r') as f :
            header = f.readline().strip()
        col_offset = 1 if 'index' in header else 0
        indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in fnames])
        oma, ranks = np.concatenate([
                                     np.loadtxt(fname, usecols=(col_offset+0, col_offset+1))
                                     for fname in fnames
                                    ], axis=0).T

        _, where_unique = np.unique(indices, return_index=True)
        oma = oma[where_unique]
        ranks = ranks[where_unique]

        assert len(ranks) == Nsamples

        # filter out failures (shouldn't be many)
        oma = oma[oma>0]
        ranks = ranks[ranks>0]

        label = make_label(run_hash, run_info)
        ax_qq.hist(oma, bins=edges_qq, histtype='step', cumulative=False, density=False,
                   label=label)

        ax_ra.hist(ranks, bins=edges_ra, histtype='step', density=False,
                   label=label)

    ax_qq.set_xlim(0, 1)
    ax_qq.set_ylim(0, None)

    ax_ra.set_xlim(0, 1)
    ax_ra.set_ylim(0, None)

    nsigma = 2
    for a, Nb in zip([ax_qq, ax_ra, ], [Nbins_qq, Nbins_ra, ]) :
        avg = Nsamples / Nb
        a.fill_between([0, 1], avg-nsigma*np.sqrt(avg), avg+nsigma*np.sqrt(avg), alpha=0.3, color='grey')

    if qq_legend_kwargs is None :
        qq_legend_kwargs = dict(loc='upper left', ncol=3)
    ax_qq.legend(**qq_legend_kwargs, frameon=False)
    ax_qq.set_xlabel('confidence level')
    ax_qq.set_ylabel('number of chains')

    if ra_legend_kwargs is None :
        ra_legend_kwargs = dict(loc='lower left', ncol=3)
    ax_ra.legend(**ra_legend_kwargs, frameon=False)
    ax_ra.set_xlabel('fractional rank of true $S_8$ within MCMC samples')
    ax_ra.set_ylabel('number of chains')

    return fig_qq, fig_ra
