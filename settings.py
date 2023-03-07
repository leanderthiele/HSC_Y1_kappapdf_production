# smoothing scales are
# 0 --  1 arcmin
# 1 --  2 arcmin
# 2 --  5 arcmin
# 3 --  7 arcmin
# 4 --  8 arcmin
# 5 -- 10 arcmin
# 6 -- 15 arcmin
# 7 -- 25 arcmin

# The cov_mode setting works as follows:
# fixed ... block is kept fixed at the fiducial point
# gpr   ... block's inverse is emulated
# scale ... block's corr is kept fixed at fiducial point
#           but scaling with sigma (from emulator) is performed
# In all cases, we keep the cross-correlations between different
# summary statistics fixed at their values at the fiducial point.

S = {
     # pdf settings, if not included we do not use pdf
     'pdf': {
             'unitstd': True,
             'log': True,
             'zs': [1, 2, 3, ],
             'smooth': [2, 5, ],
             'rebin': 1,
             'low_cut': 3,
             'high_cut': 0,
             'delete': 9, # delete one bin to prevent ill-conditioned covariance from sum constraint
             'cov_mode': 'gpr',
            },
     
     # power spectrum settings, if not included we do not use power spectrum
     'ps': {
            'rebin': 1,
            'zs': [0, ],
            'low_cut': 4,
            'high_cut': 6,
            'cov_mode': 'fixed',
           },

     # this is a bit ugly but who cares (special case when MOPED in joint mode)
     # probably never used...
     'joint': { 'cov_mode': 'gpr', },
     
     # moped settings
     'moped': {
               'deriv_mode': 'gpr', # either lstsq_N or gpr, should be about equivalent
               'apply_to': ['pdf', ], # list means separate and then concatenated,
                                      # 'joint' would mean on joint data vector
              },
    }

import hashlib
IDENT = hashlib.md5(f'{S}'.encode('utf-8')).hexdigest()
