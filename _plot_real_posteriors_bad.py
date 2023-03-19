from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

bad_runs = {
            'edc9498884171615f40fbe07e99aee90': 'cov fixed',
           }

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(bad_runs, make_label)

savefig(fig, 'bad_real_posteriors')
