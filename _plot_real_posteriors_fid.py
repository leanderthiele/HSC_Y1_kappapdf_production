from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

fid_runs = {
            # '22927c646f516731cb39e41daab9e6e5': '$C_\ell^{\kappa\kappa}$',
            'd1b2a20ab66f58289a9b1509ceebf44a': 'PDF', # 5, 8, 10 arcmin
            'e288639850d10d67ee3c493cf37ca5d6': 'PDF 5->4arcmin', # 4, 8, 10 arcmin
            # '4115edf8bc62c922be1c8f4eaac8345b': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 8, 10 arcmin
           }

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(fid_runs, make_label, have_numbers=False)

savefig(fig, 'real_posteriors')
