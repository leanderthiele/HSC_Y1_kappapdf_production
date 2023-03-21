from glob import glob

from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

from settings import CHAIN_ROOT
from data import Data
from _plot_style import *

run_hash = '9d56790a0f55a6885899ec32284b91bd'

delta = 0.02 # this is the window we consider

coverage_fnames = glob(f'{CHAIN_ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
# figure out if this is one of the old runs where we didn't have the index information
with open(coverage_fnames[0], 'r') as f :
    header = f.readline().strip()
col_offset = 1 if 'index' in header else 0
indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in coverage_fnames])
ranks = np.concatenate([np.loadtxt(fname, usecols=(col_offset+1,)) for fname in coverage_fnames], axis=0)
_, where_unique = np.unique(indices, return_index=True)
indices = indices[where_unique]
ranks = ranks[where_unique]

select = (ranks > 0)
indices = indices[select]
ranks = ranks[select]

select = (ranks < delta) | (ranks > 1-delta)
indices = indices[select]
ranks = ranks[select]

# select for reasonable S8
data = Data()
true_S8 = data.get_cosmo('cosmo_varied')[:, 0]
nseeds = data.get_nseeds('cosmo_varied')
selector = lambda x: 0.75<x<0.85
select = np.array([selector(true_S8[idx // nseeds]) for idx in indices], dtype=bool)
indices = indices[select]
ranks = ranks[select]

print(f'Have {len(indices)} candidates')

rng = np.random.default_rng()
p = min(( 20/len(indices), 1 ))
select = rng.choice([True, False], len(indices), p=[p, 1-p])
indices = indices[select]
ranks = ranks[select]

prior = {
         'S8': (0.50, 1.00),
         'Om': (0.20, 0.40),
        }

fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
ax_S8 = ax[0]
ax_Om = ax[1]

for idx in tqdm(indices) :
    
    chain_fname = f'{CHAIN_ROOT}/cosmo_varied_{run_hash}/chain_{idx}.npz'
    with np.load(chain_fname) as f :
        S8, Om = f['chain'].reshape(-1, 2).T

    for y, a, p in zip([S8, Om, ], [ax_S8, ax_Om, ], prior.values()) :
        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01).fit(y.reshape(-1, 1))
        x = np.linspace(*p, num=100)
        logh = kde.score_samples(x.reshape(-1, 1))
        logh -= np.max(logh)
        h = np.exp(logh)

        a.plot(x, h)

for a, p, label in zip([ax_S8, ax_Om, ], prior.values(), ['$S_8$', '$\Omega_m$', ]) :
    a.set_xlim(*p)
    a.set_ylim(0, None)
    a.set_xlabel(label)
    a.set_yticks([])

savefig(fig, 'catastrophes')
