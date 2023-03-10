
stat_runs = {
             # not perfect but ok
             'befab23d6ee10fe971a5ad7118957c9c': 'baseline PDF',
             # clearly overconfident
             # '3e14c0d1a34c1aa19ab78949396014de': 'cov_mode=gpr_scale',
             # clearly overconfident
             # '45b6a5961cb86a0e7690fc6919c9bf8e': 'cov_mode=fixed',
             # somewhat overconfident
             # 'd9d31391c8a306a481a4a26ce07969d2': 'baseline PDF+PS',
             # confirms that the (rare) catastrophic failures come from the power spectrum
             '68c282161ba83a2267303b9ea1500119': 'PS only, cov_mode=fixed',
             # better than d9d3 but still overconfident
             'b8f4e40091ee24e646bb879d225865f6': 'PDF+PS, PS cov_mode=fixed',
            }


def stat_make_label (run_hash, run_info) :
    return f'$\\tt{{ {run_hash[:4]} }}$: {run_info}'
