from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

bad_runs = {
            'fd47089b3f34889e50653bbb4ebeff98': 'baseline',
            'edc9498884171615f40fbe07e99aee90': 'cov fixed',
            '4eaaed4780c3db75fbdffd31d1aac885': 'no compression',
            '7e4e8c4eb70d75d6bcfb03d92cca7abd': 'no log',
           }

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(bad_runs, make_label, have_numbers=False)

savefig(fig, 'bad_real_posteriors')
