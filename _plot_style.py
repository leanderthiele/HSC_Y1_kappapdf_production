# Import this to get a uniform style

from matplotlib import pyplot as plt
from itertools import cycle

if True :
    plt.style.use('dark_background')
    black = 'white'
    white = 'black'
    bg_ident = '_dark_bg'
else :
    black = 'black'
    white = 'white'
    bg_ident = ''

if True :
    fmt = 'png'
    kwargs = dict(transparent=True, dpi=400)
else :
    fmt = 'pdf'
    kwargs = dict()

# plt.rcParams.update({'font.size': 20})
# plt.rc('text', usetex=True)

def savefig (fig, name) :
    outdir = '.'

    fig.savefig(f'{outdir}/_plot{bg_ident}_{name}.{fmt}', bbox_inches='tight', **kwargs)

# type : list
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
default_linestyles = ['-', '--', '-.', ':']
default_markers = ['o', 'x', 's', ]
