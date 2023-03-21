from _plot_style import *
from _plot_calibration import PlotCalibration

fid_runs = {
             'fd47089b3f34889e50653bbb4ebeff98': 'PDF',
             '22927c646f516731cb39e41daab9e6e5': '$C_\ell^{\kappa\kappa}$',
             '9d56790a0f55a6885899ec32284b91bd': 'PDF+$C_\ell^{\kappa\kappa}$',
            }


def fid_make_label (run_hash, run_info) :
    return run_info

fig_qq, fig_ra = PlotCalibration(fid_runs, fid_make_label)
savefig(fig_qq, 'calibration_qq')
savefig(fig_ra, 'calibration_ranks')
