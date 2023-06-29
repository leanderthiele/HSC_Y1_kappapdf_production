from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from settings import CHAIN_ROOT
from _plot_style import *

pdf_hash = 'd1b2a20ab66f58289a9b1509ceebf44a'
ps_hash = '22927c646f516731cb39e41daab9e6e5'

def get_from_hash (h) :
    fnames = glob(f'{CHAIN_ROOT}/fiducial_{h}/bias_data_*.dat')
    indices = np.concatenate([np.loadtxt(fname, usecols=0) for fname in fnames])
    means = np.concatenate([np.loadtxt(fname, usecols=1) for fname in fnames])
    medians = np.concatenate([np.loadtxt(fname, usecols=3) for fname in fnames])
    _, uniq_idx = np.unique(indices, return_index=True)
    indices = indices[uniq_idx]
    means = means[uniq_idx]
    medians = medians[uniq_idx]
    return indices, means, medians

def get_from_idx (i, x, i_b) :
    return np.array([x[i==ii] for ii in i_b])

i_pdf, mean_pdf, median_pdf = get_from_hash(pdf_hash)
i_ps,  mean_ps,  median_ps  = get_from_hash(ps_hash)

i_both = np.array(list(set.intersection(set(i_pdf), set(i_ps))))

i_pdf = get_from_idx(i_pdf, i_pdf, i_both)
mean_pdf = get_from_idx(i_pdf, mean_pdf, i_both)
median_pdf = get_from_idx(i_pdf, median_pdf, i_both)

i_ps = get_from_idx(i_ps, i_ps, i_both)
mean_ps = get_from_idx(i_ps, mean_ps, i_both)
median_ps = get_from_idx(i_ps, median_ps, i_both)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(median_ps, median_pdf)

ax.axline((0, 0), slope=1, linestyle='dashed', color=black)

ax.set_xlabel('median from $C_\ell^{\kappa\kappa}$')
ax.set_ylabel('median from PDF')

savefig(fig, 'fidscatter')
