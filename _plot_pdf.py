import numpy as np
from matplotlib import pyplot as plt

from settings import S
from _plot_style import *

if 'ps' in S :
    del S['ps']

S['pdf'] = {
            'unitstd': True,
            'log': True,
            'zs': [2, ],
            'smooth': [2, ],
            'rebin': 1,
            'low_cut': 0,
            'high_cut': 0,
            'delete': None,
           }

from data import Data
data = Data()

kappa_edges = np.linspace(-4, 4, num=20)
kappa = 0.5 * (kappa_edges[1:] + kappa_edges[:-1])

all_fid = data.get_datavec('fiducial')
d_fid = np.mean(all_fid, axis=0)
d_hsc = data.get_datavec('real')[0]
sigma = np.std(all_fid, axis=0)

# make it zero at the origin
offset = d_fid[9]
d_fid -= offset
d_hsc -= offset

gaussian = -0.5 * kappa**2

fig, ax = plt.subplots(figsize=(5, 5))

# currently unused, could be used to indicate cuts but I feel the figure
# would be to busy for an introductory one
cut = [0, 1, 2, 9, ]
where_cut = np.array([ii in cut for ii in range(19)], dtype=bool)

ax.plot(kappa, d_fid, label='fiducial simulations')
if False :
    ax.errorbar(kappa, d_hsc, yerr=sigma, label='HSC Y1',
                linestyle='none', marker='o')
else :
    l = ax.errorbar(kappa[~where_cut], d_hsc[~where_cut], yerr=sigma[~where_cut],
                    label='HSC Y1', linestyle='none', marker='o')
    color = plt.getp(l[0], 'color')
    ax.errorbar(kappa[where_cut], d_hsc[where_cut], yerr=sigma[where_cut],
                linestyle='none', marker='o',
                markeredgecolor=color, ecolor=color, fillstyle='full', markerfacecolor=white)
ax.plot(kappa, gaussian, label='Gaussian',
        color=black)

ax.set_xlim(-4, 4)
ax.set_yticks([0, -3, -6, -9])
ax.set_xlabel('$\kappa / \sigma(\kappa)$')
ax.set_ylabel('$\log {\sf PDF}(\kappa) + {\sf const}$')
ax.legend(frameon=False, loc='lower center')

savefig(fig, 'pdf')
