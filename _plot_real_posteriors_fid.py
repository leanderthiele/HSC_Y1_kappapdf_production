from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

fid_runs = {
            '22927c646f516731cb39e41daab9e6e5': '$C_\ell^{\kappa\kappa}$',
            'fd47089b3f34889e50653bbb4ebeff98': 'PDF', # 5, 7, 10 arcmin
            '9d56790a0f55a6885899ec32284b91bd': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 10 arcmin
           }

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(fid_runs, make_label)

savefig(fig, 'real_posteriors')
