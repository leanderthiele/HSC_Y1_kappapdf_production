from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from settings import CHAIN_ROOT
from _plot_style import *

USE_FID = False

pdf_hash = 'd1b2a20ab66f58289a9b1509ceebf44a'
ps_hash = '22927c646f516731cb39e41daab9e6e5'

# pdf - ps medians
delta = 0.0344

def get_from_hash (h) :
    fnames = glob(f'{CHAIN_ROOT}/{"fiducial" if USE_FID else "cosmo_varied"}_{h}/'\
                  f'{"bias" if USE_FID else "coverage"}_data*.dat')
    indices = np.concatenate([np.loadtxt(fname, usecols=0) for fname in fnames])
    means = np.concatenate([np.loadtxt(fname, usecols=1 if USE_FID else 4) for fname in fnames])
    medians = np.concatenate([np.loadtxt(fname, usecols=3 if USE_FID else 6) for fname in fnames])
    _, uniq_idx = np.unique(indices, return_index=True)
    indices = indices[uniq_idx]
    means = means[uniq_idx]
    medians = medians[uniq_idx]
    return indices, means, medians

def get_from_idx (i, x, i_b) :
    foo = np.array([ii in i_b for ii in i])
    return x[foo]

i_pdf, mean_pdf, median_pdf = get_from_hash(pdf_hash)
i_ps,  mean_ps,  median_ps  = get_from_hash(ps_hash)

i_both = np.array(list(set.intersection(set(i_pdf), set(i_ps))))

mean_pdf = get_from_idx(i_pdf, mean_pdf, i_both)
median_pdf = get_from_idx(i_pdf, median_pdf, i_both)
i_pdf = get_from_idx(i_pdf, i_pdf, i_both)

mean_ps = get_from_idx(i_ps, mean_ps, i_both)
median_ps = get_from_idx(i_ps, median_ps, i_both)
i_ps = get_from_idx(i_ps, i_ps, i_both)

print(i_pdf)
print(i_ps)

n = np.count_nonzero((median_pdf-median_ps)>delta)
print(f'{n/len(median_pdf)*100} percent have larger difference in this direction')
n = np.count_nonzero(np.fabs(median_pdf-median_ps)>delta)
print(f'{n/len(median_pdf)*100} percent have larger difference in both directions')
n = np.count_nonzero(median_pdf>median_ps)
print(f'{n/len(median_pdf)*100} percent have larger difference in both directions')

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(median_ps, median_pdf)

ax.axline((0, 0), slope=1, linestyle='dashed', color=black)
ax.axline((0, delta), slope=1, linestyle='dashed', color=black)

ax.set_xlabel('median from $C_\ell^{\kappa\kappa}$')
ax.set_ylabel('median from PDF')
if USE_FID :
    ax.set_xlim(0.6, 0.95)
    ax.set_ylim(0.6, 0.95)
else :
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    

savefig(fig, 'fidscatter')
