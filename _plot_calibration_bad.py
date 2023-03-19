from _plot_style import *
from _plot_calibration import PlotCalibration

bad_runs = {
            'edc9498884171615f40fbe07e99aee90': 'cov fixed',
           }

def bad_make_label (run_hash, run_info) :
    return run_info

fig_qq, fig_ra = PlotCalibration(bad_runs, bad_make_label)
savefig(fig_qq, 'miscalibration_qq')
savefig(fig_ra, 'miscalibration_ranks')

