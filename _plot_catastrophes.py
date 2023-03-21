from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from settings import CHAIN_ROOT
from _plot_style import *

run_hash = '9d56790a0f55a6885899ec32284b91bd'

delta = 0.05 # this is the window we consider

coverage_fnames = glob(f'{CHAIN_ROOT}/cosmo_varied_{run_hash}/coverage_data_[0-9]*.dat')
# figure out if this is one of the old runs where we didn't have the index information
with open(fnames[0], 'r') as f :
    header = f.readline().strip()
col_offset = 1 if 'index' in header else 0
indices = np.concatenate([np.loadtxt(fname, usecols=(0,), dtype=int) for fname in coverage_fnames])
ranks = np.concatenate([np.loadtxt(fname, usecols=(col_offset+1,)) for fname in coverage_fnames], axis=0)
_, where_unique = np.unique(indices, return_index=True)
indices = indices[where_unique]
ranks = ranks[where_unique]

select = (ranks < delta) | (ranks > 1-delta)
indices = indices[select]
ranks = ranks[select]

print(f'Have {len(indices)} candidates')
