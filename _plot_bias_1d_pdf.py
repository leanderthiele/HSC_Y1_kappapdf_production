from _plot_style import *
from _plot_bias_1d import PlotBias1D

runs = {
        'fd47089b3f34889e50653bbb4ebeff98': \
             {
              'fiducial': 'fiducial',
              'mbias/mbias_plus': '$\Delta m = +1\,\%$',
              'mbias/mbias_minus': '$\Delta m = -1\,\%$',
              'fiducial-baryon': 'baryons',
              'photoz/frankenz': '${\\tt frankenz}$',
              'photoz/mizuki': '${\\tt mizuki}$',
              'fiducial-IA/032_simple': '$A_{\sf IA} = -0.32$',
              'fiducial-IA/118_simple': '$A_{\sf IA} = 1.18$',
             },
       }

def make_label (run_hash, bias_info) :
    return bias_info

fig, fig_bars = PlotBias1D(runs, make_label)

savefig(fig, 'bias_1d_pdf')
savefig(fig_bars, 'bias_bars_pdf')
