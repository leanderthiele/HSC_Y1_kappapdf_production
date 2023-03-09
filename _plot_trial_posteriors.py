from glob import glob
import re
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt

import corner

from data import Data
from _plot_stat_runs import *
from _plot_style import *

NROWS = 3
NCOLS = 3

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
assert len(avail_indices) >= NROWS * NCOLS
