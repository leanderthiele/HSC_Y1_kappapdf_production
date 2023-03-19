from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

fid_runs = {
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

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(fid_runs, make_label)

savefig(fig, 'real_posteriors')
