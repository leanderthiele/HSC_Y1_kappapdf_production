from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

bad_runs = {
            '65fd48ca07f7aa3b36504f765f9dbf2d': 'PDF covariance fixed',
           }

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(fid_runs, make_label)

savefig(fig, 'bad_real_posteriors')
