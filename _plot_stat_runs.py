
stat_runs = {
             'fd47089b3f34889e50653bbb4ebeff98': 'PDF',
             '22927c646f516731cb39e41daab9e6e5': '$C_\ell^{\kappa\kappa}$',
             '9d56790a0f55a6885899ec32284b91bd': 'PDF+$C_\ell^{\kappa\kappa}$',
            }


def stat_make_label (run_hash, run_info) :
#    return f'$\\tt{{ {run_hash[:4]} }}$: {run_info}'
    return run_info
