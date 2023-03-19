from _plot_style import *
from _plot_calibration import PlotCalibration

bad_runs = {
            '65fd48ca07f7aa3b36504f765f9dbf2d': 'PDF covariance fixed',
           }

def bad_make_label (run_hash, run_info) :
    return run_info

fig_qq, fig_ra = PlotCalibration(bad_runs, bad_make_label)
savefig(fig_qq, 'miscalibration_qq')
savefig(fig_ra, 'miscalibration_ranks')

