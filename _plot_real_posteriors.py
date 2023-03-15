import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

from _plot_style import *

real_runs = {
             '68c282161ba83a2267303b9ea1500119': '$C_\ell^{\kappa\kappa}$, ours',
            }

def make_label (run_hash, run_info) :
    return run_info

fig, ax = plt.subplots(figsize=(5,5))

prior = (0.45, 1.05)
x = np.linspace(*prior, num=500)

for run_hash, run_info in real_runs.items() :
    
    with np.load(f'real_chain_{run_hash}.npz') as f :
        chain = f['chain']
    S8 = chain.reshape(-1, chain.shape[-1])[:, 0]

    avg = np.mean(S8)
    std = np.std(S8)
    print(f'{run_hash[:4]}: {avg:.3f} +- {std:.3f}')
    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01).fit(S8.reshape(-1, 1))
    logh = kde.score_samples(x.reshape(-1, 1))
    logh -= np.max(logh)
    h = np.exp(logh)

    label = make_label(run_hash, run_info)
    ax.plot(x, h, label=label)

ax.legend(loc='upper left', frameon=False)
ax.set_xlim(*prior)
ax.set_ylim(0, None)

savefig(fig, 'real_posteriors')
