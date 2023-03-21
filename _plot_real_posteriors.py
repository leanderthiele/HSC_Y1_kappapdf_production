import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import basinhopping
from sklearn.neighbors import KernelDensity

from _plot_style import *

compare = {
           'Hikage+2019 $C_\ell^{\kappa\kappa}$': (0.78, (0.033, 0.030)),
           'Hamana+2020 $\\xi_{\\pm}(\\theta)$': (0.823, (0.028, 0.032)),
          }

COMPARE_STYLE = 'bars' # can be 'bars', 'bands', 'curves'

prior = (0.50, 1.00)
x = np.linspace(*prior, num=500)

def PlotRealPosteriors (runs, make_label, have_numbers=True, all_have_Cl=False) :

    fig, ax = plt.subplots(figsize=(5,3))

    for run_hash, run_info in runs.items() :

        label = make_label(run_hash, run_info)
        
        with np.load(f'real_chain_{run_hash}.npz') as f :
            chain = f['chain']
        S8 = chain.reshape(-1, chain.shape[-1])[:, 0]

        avg = np.mean(S8)
        std = np.std(S8)
    #    print(f'{run_hash[:4]}: {avg:.3f} +- {std:.3f}')
        
        kde = KernelDensity(kernel='epanechnikov',
                            bandwidth=0.03 if 'C_\ell' not in run_info and not all_have_Cl else 0.01)\
                    .fit(S8.reshape(-1, 1))

        if have_numbers :
            nll = lambda x_ : -kde.score_samples(np.array([[x_]]).reshape(-1, 1))
            sln = basinhopping(nll, 0.8, T=0.1, niter=10,
                               minimizer_kwargs={'bounds': [(np.min(S8)+0.02, np.max(S8)-0.02), ]})
            S8_map = sln.x.item()
            S8_hi = np.quantile(S8, 0.84)
            S8_lo = np.quantile(S8, 0.16)
            delta_hi = S8_hi - S8_map
            delta_lo = S8_map - S8_lo
            print(f'{run_hash[:4]}: S8 = {S8_map:.4f} +{delta_hi:.4f} -{delta_lo:.4f} [{label}]')
            label = f'{label}, ${S8_map:.3f}^{{+ {delta_hi:.3f} }}_{{- {delta_lo:.3f} }}$'

        logh = kde.score_samples(x.reshape(-1, 1))
        logh -= np.max(logh)
        h = np.exp(logh)

        ax.plot(x, h, label=label)

    for ii, (label, (avg, errs)) in enumerate(compare.items()) :
        if COMPARE_STYLE == 'curves' :
            std = sum(errs) / len(errs)
            ax.plot(x, np.exp(-0.5*((x-avg)/std)**2), color=black, linestyle=default_linestyles[ii],
                    label=label)
        elif COMPARE_STYLE == 'bars' :
            y = 1.05 + 0.1*ii
            ax.errorbar([avg, ], [y, ], xerr=([errs[0], ], [errs[1], ]), marker='o', color=black, markersize=3, capsize=3)
            # orient = 'r' if avg<np.mean(np.array([m for m, _ in compare.values()])) else 'l'
            orient = 'r'
            ax.text(avg+(-1 if orient=='l' else 1)*(errs[0 if orient=='l' else 1]+0.01), y, label, fontsize='x-small',
                    ha='left' if orient=='r' else 'right', va='center',
                    transform=ax.transData)

    ax.legend(loc='upper left', frameon=False, labelspacing=0.7)
    ax.set_xlim(*prior)
    ax.set_ylim(0, None)
    ax.set_xlabel('$S_8 = \sigma_8 \sqrt{\Omega_m/0.3}$')
    ax.set_yticks([])
    ax.set_ylabel('posterior $p(S_8)$')

    return fig
