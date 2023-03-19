import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import basinhopping
from sklearn.neighbors import KernelDensity

from _plot_style import *

real_runs = {
             # '68c282161ba83a2267303b9ea1500119': '$C_\ell^{\kappa\kappa}$, ours singlez',
             # '781d7e9046306def06188bd946264d10': '$C_\ell^{\kappa\kappa}$', # old prior
             '22927c646f516731cb39e41daab9e6e5': '$C_\ell^{\kappa\kappa}$',

             # PDF are all tomographic here, with last z bin cut
             # 'befab23d6ee10fe971a5ad7118957c9c': 'PDF', # 5, 10 arcmin
             # '82b00bf14b9540829377a65c9d552683': 'PDF', # 5, 7, 10 arcmin, old prior
             'fd47089b3f34889e50653bbb4ebeff98': 'PDF', # 5, 7, 10 arcmin

             # '7f609754f21d842b69707dd22d47d0b1': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 10 arcmin
             # 'e8365fe775638de85ed8010d2d23f1dc': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 10 arcmin
             # 'd6f1a70dc7eae4b4f4e19dafdee50b95': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 8, 10 arcmin
             '9d56790a0f55a6885899ec32284b91bd': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 10 arcmin
            }

compare = {
           'Hikage+2019 $C_\ell^{\kappa\kappa}$': (0.78, 0.0315),
           'Hamana+2020 $\\xi_{\\pm}(\\theta)$': (0.823, 0.030),
          }

COMPARE_STYLE = 'bars' # can be 'bars', 'bands', 'curves'

def make_label (run_hash, run_info) :
    return run_info

fig, ax = plt.subplots(figsize=(5,3))

prior = (0.50, 1.00)
x = np.linspace(*prior, num=500)

for run_hash, run_info in real_runs.items() :

    label = make_label(run_hash, run_info)
    
    with np.load(f'real_chain_{run_hash}.npz') as f :
        chain = f['chain']
    S8 = chain.reshape(-1, chain.shape[-1])[:, 0]

    avg = np.mean(S8)
    std = np.std(S8)
#    print(f'{run_hash[:4]}: {avg:.3f} +- {std:.3f}')
    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.03 if 'C_\ell' not in run_info else 0.01)\
                .fit(S8.reshape(-1, 1))

    nll = lambda x_ : -kde.score_samples(np.array([[x_]]).reshape(-1, 1))
    sln = basinhopping(nll, 0.8, T=0.1, niter=10, minimizer_kwargs={'bounds': [(0.7, 0.9), ]})
    S8_map = sln.x.item()
    S8_hi = np.quantile(S8, 0.84)
    S8_lo = np.quantile(S8, 0.16)
    delta_hi = S8_hi - S8_map
    delta_lo = S8_map - S8_lo
    print(f'{run_hash[:4]}: S8 = {S8_map:.4f} +{delta_hi:.4f} -{delta_lo:.4f} [{label}]')

    logh = kde.score_samples(x.reshape(-1, 1))
    logh -= np.max(logh)
    h = np.exp(logh)

    new_label = f'{label}, ${S8_map:.3f}^{{+ {delta_hi:.3f} }}_{{- {delta_lo:.3f} }}$'

    ax.plot(x, h, label=new_label)

for ii, (label, (avg, std)) in enumerate(compare.items()) :
    if COMPARE_STYLE == 'curves' :
        ax.plot(x, np.exp(-0.5*((x-avg)/std)**2), color=black, linestyle=default_linestyles[ii],
                label=label)
    elif COMPARE_STYLE == 'bars' :
        y = 1.05 + 0.1*ii
        ax.errorbar([avg, ], [y, ], xerr=std, marker='o', color=black, markersize=3, capsize=3)
        # orient = 'r' if avg<np.mean(np.array([m for m, _ in compare.values()])) else 'l'
        orient = 'r'
        ax.text(avg+(-1 if orient=='l' else 1)*(std+0.01), y, label, fontsize='x-small',
                ha='left' if orient=='r' else 'right', va='center',
                transform=ax.transData)

ax.legend(loc='upper left', frameon=False, labelspacing=0.7)
ax.set_xlim(*prior)
ax.set_ylim(0, None)
ax.set_xlabel('$S_8 = \sigma_8 \sqrt{\Omega_m/0.3}$')
ax.set_yticks([])
ax.set_ylabel('posterior $p(S_8)$')

savefig(fig, 'real_posteriors')
