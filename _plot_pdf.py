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

d_fid = np.mean(data.get_datavec('fiducial'), axis=0)
print(d_fid.shape)
d_hsc = data.get_datavec('real')

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(kappa, d_fid, label='fiducial simulations')
ax.plot(kappa, d_hsc, label='HSC Y1',
        linestyle='none', marker='o')

ax.set_xlabel('$\kappa / \sigma(\kappa)$')
ax.set_ylabel('$\log {\sf PDF}(\kappa) + {\sf const}$')

savefig(fig, 'pdf')
