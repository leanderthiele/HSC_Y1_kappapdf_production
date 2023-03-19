from _plot_style import *
from _plot_real_posteriors import PlotRealPosteriors

runs = {
        '9d56790a0f55a6885899ec32284b91bd': 'baseline',
        '30e661b4d6971241ac0d8ebc14fc2497': 'no z1',
        '6036999f493df78b2b131afacf3567de': 'no z3',
        '6ad28c1981d09537eae2af5d21534382': 'with z4',
        '96ddc803979f67fe994b9e3149a68e52': 'single-$z$',
        '23efd39155c09883b519fb66eb149688': 'no $5\,{\sf arcmin}$',
        '5d3408187c60e5a444ac55999f9da9e8': 'no highest $\kappa$',
       }

def make_label (run_hash, run_info) :
    return run_info

fig = PlotRealPosteriors(runs, make_label, have_numbers=False, all_have_Cl=True)

savefig(fig, 'cut_real_posteriors')
