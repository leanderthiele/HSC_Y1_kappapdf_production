import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

from _plot_style import *

real_runs = {
             # '68c282161ba83a2267303b9ea1500119': '$C_\ell^{\kappa\kappa}$, ours singlez',
             '781d7e9046306def06188bd946264d10': '$C_\ell^{\kappa\kappa}$, ours tomography',

             # PDF are all tomographic here, with last z bin cut
             # 'befab23d6ee10fe971a5ad7118957c9c': 'PDF', # 5, 10 arcmin
             '82b00bf14b9540829377a65c9d552683': 'PDF', # 5, 7, 10 arcmin
             # '7f609754f21d842b69707dd22d47d0b1': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 10 arcmin
             'e8365fe775638de85ed8010d2d23f1dc': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 10 arcmin
             # 'd6f1a70dc7eae4b4f4e19dafdee50b95': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 8, 10 arcmin
            }

def make_label (run_hash, run_info) :
    return run_info

fig, ax = plt.subplots(figsize=(5,5))

prior = (0.50, 1.00)
x = np.linspace(*prior, num=500)

for run_hash, run_info in real_runs.items() :
    
    with np.load(f'real_chain_{run_hash}.npz') as f :
        chain = f['chain']
    S8 = chain.reshape(-1, chain.shape[-1])[:, 0]

    avg = np.mean(S8)
    std = np.std(S8)
    print(f'{run_hash[:4]}: {avg:.3f} +- {std:.3f}')
    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.05 if 'C_\ell' not in run_info else 0.01)\
                .fit(S8.reshape(-1, 1))
    logh = kde.score_samples(x.reshape(-1, 1))
    logh -= np.max(logh)
    h = np.exp(logh)

    label = make_label(run_hash, run_info)
    ax.plot(x, h, label=label)

ax.plot(x, np.exp(-0.5*((x-0.780)/0.0315)**2), color=black, linestyle='dashed',
        label='Hikage+2019 $C_\ell^{\kappa\kappa}$')
ax.plot(x, np.exp(-0.5*((x-0.823)/0.030)**2), color=black, linestyle='dotted',
        label='Hamana+2020 $\\xi_{\\pm}(\\theta)$')

ax.legend(loc='upper left', frameon=False)
ax.set_xlim(*prior)
ax.set_ylim(0, None)
ax.set_xlabel('$S_8 = \sigma_8 \sqrt{\Omega_m/0.3}$')
ax.set_yticks([])

savefig(fig, 'real_posteriors')
