import numpy as np
from matplotlib import pyplot as plt

from data import Data
from _plot_style import *

S8, Om = Data().get_cosmo('cosmo_varied').T

prior = { 'S8': (0.45, 1.05), 'Om': (0.20, 0.40), }

fig, ax = plt.subplots(figsize=(5,2.5))

ax.plot(Om, S8, linestyle='none', marker='o')
rect = plt.Rectangle( (prior['Om'][0], prior['S8'][0]),
                     prior['Om'][1]-prior['Om'][0], prior['S8'][1]-prior['S8'][0],
                     fill=False, edgecolor=black)
ax.add_patch(rect)
ax.text(prior['Om'][1], prior['S8'][1], 'prior',
        va='bottom', ha='right')

ax.set_xlabel('$\Omega_m$')
ax.set_ylabel('$S_8$')

savefig(fig, 'sims')
