import numpy as np
from matplotlib import pyplot as plt

import corner

from data import Data
from _plot_stat_runs import *
from _plot_style import *
from _plot_get_test_trials import GetTestTrials

NROWS = 2
NCOLS = 2

NAX = NROWS * NCOLS

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# do some selection
data = Data()
theta_sims = data.get_cosmo('cosmo_varied')
use_indices = GetTestTrials(stat_runs.keys(), NAX)

fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(3*NROWS,3*NCOLS),
                       gridspec_kw=dict(hspace=0, wspace=0))
ax = ax.flatten()
gridspec = ax[0].get_subplotspec().get_gridspec()

# some utilities used below
close = lambda x, y : abs(x-y)<1e-5
S8_text = lambda avg, std=None : f'$S_8 = {avg:.2f}\pm{std:.2f}$' if std is not None \
                                 else f'$S_8^* = {avg:.2f}$'
S8_text_kwargs = lambda a : dict(ha='left', va='top', transform=a.transAxes)

for ii, (a, idx) in enumerate(zip(ax, use_indices)) :

    # need to do some tricks here to make corner cooperate
    row, col = divmod(ii, NCOLS)
    a.remove()
    subfig = fig.add_subfigure(gridspec[row, col])
    new_ax = subfig.subplots(2, 2)
    
    # put the true value in
    cosmo_idx = idx//Data.NSEEDS['cosmo_varied']
    assert cosmo_idx in allowed_cosmo_indices
    true_S8, true_Om = theta_sims[cosmo_idx]
    new_ax[0, 0].axvline(true_S8, color=black)
    new_ax[1, 0].axvline(true_S8, color=black)
    new_ax[1, 0].axhline(true_Om, color=black)
    new_ax[1, 1].axvline(true_Om, color=black)

    new_ax[0, 1].text(0, 1, S8_text(true_S8), color=black, **S8_text_kwargs(new_ax[0, 1]))

    for ii, (run_hash, run_info) in enumerate(stat_runs.items()) :
        chain_fname = f'{ROOT}/cosmo_varied_{run_hash}/chain_{idx}.npz'
        with np.load(chain_fname) as f :
            chain = f['chain'].reshape(-1, 2)
            _true_theta = f['true_theta']
        assert all(close(t1, t2) for t1, t2 in zip(_true_theta, [true_S8, true_Om]))

        corner.corner(chain, labels=['$S_8$', '$\Omega_m$', ],
                      range=[(0.45, 1.05), (0.20, 0.40)],
                      color=default_colors[ii],
                      fig=subfig,
                      plot_datapoints=False, plot_density=False, no_fill_contours=True,
                      levels=1 - np.exp(-0.5 * np.array([1])**2)
                     )
        
        new_ax[0, 1].text(0, 1-0.14*(ii+1), S8_text(np.mean(chain[:,0]), np.std(chain[:,0])),
                          color=default_colors[ii], **S8_text_kwargs(new_ax[0, 1]))


savefig(fig, 'trial_posteriors')
