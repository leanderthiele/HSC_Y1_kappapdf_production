from glob import glob
import re
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt

import corner

from data import Data
from _plot_stat_runs import *
from _plot_style import *

NROWS = 2
NCOLS = 2

NAX = NROWS * NCOLS

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# only consider runs with these true values
restrict = {
            'S8': (0.75, 0.85),
            'Om': (0.25, 0.35),
           }

# get the cosmology indices that fall into restrict
data = Data()
theta_sims = data.get_cosmo('cosmo_varied')
allowed_cosmo_indices = []
in_interval = lambda x, t: t[0] <= x <= t[1]
for ii, theta in enumerate(theta_sims) :
    if all(in_interval(t, r) for t, r in zip(theta, restrict.values())) :
        allowed_cosmo_indices.append(ii)

# get the runs that we have available
pattern = re.compile('chain_([0-9]*).npz')
all_avail_indices = []
all_avail_cosmo_indices = []
for run_hash in stat_runs.keys() :
    chain_fnames = glob(f'{ROOT}/cosmo_varied_{run_hash}/chain_[0-9]*.npz')
    avail_indices = [int(pattern.search(chain_fname)[1]) for chain_fname in chain_fnames]
    avail_cosmo_indices = [idx//Data.NSEEDS['cosmo_varied'] for idx in avail_indices]
    # filter by our restriction
    avail_indices, avail_cosmo_indices = zip(*((idx, cosmo_idx) \
                                               for idx, cosmo_idx in zip(avail_indices, avail_cosmo_indices) \
                                               if cosmo_idx in allowed_cosmo_indices))
    all_avail_indices.append(avail_indices)
    all_avail_cosmo_indices.append(avail_cosmo_indices)

# first try to get exactly the same runs, maybe we have enough
# if we have about a thougsand runs this is usually enough
avail_indices = set.intersection(*map(set, all_avail_indices))
assert len(avail_indices) >= NAX

use_indices = np.random.default_rng().choice(list(avail_indices), size=NAX, replace=False)

fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(3*NROWS,3*NCOLS),
                       gridspec_kw=dict(hspace=0, wspace=0))
ax = ax.flatten()
gridspec = ax[0].get_subplotspec().get_gridspec()

# works...
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
        
        new_ax[0, 1].text(0, 1-0.12*(ii+1), S8_text(np.mean(chain[:,0]), np.std(chain[:,0])),
                          color=default_colors[ii], **S8_text_kwargs(new_ax[0, 1]))


savefig(fig, 'trial_posteriors')
