# smoothing scales are
# 0 --  1 arcmin
# 1 --  2 arcmin
# 2 --  5 arcmin
# 3 --  7 arcmin
# 4 --  8 arcmin
# 5 -- 10 arcmin
# 6 -- 15 arcmin
# 7 -- 25 arcmin

S = {
     # source redshift bins, 0 is single-z
     'zs': [0, 1, 2, 3, 4, ],

     # pdf settings, if not included we do not use pdf
     'pdf': {
             'unitstd': True,
             'log': True,
             'smooth': [2, 5, ],
             'rebin': 1,
             'low_cut': 3,
             'high_cut': 0,
             'delete': 9, # delete one bin to prevent ill-conditioned covariance from sum constraint
            },
     
     # power spectrum settings, if not included we do not use power spectrum
     # 'ps': {
     #        'rebin': 1,
     #        'low_cut': 4,
     #        'high_cut': 6,
     #       },
     
     # moped settings
     'moped': {
               'deriv_mode': 'gpr', # either lstsq_N or gpr, should be about equivalent
               'apply_to': ['pdf', ], # list means separate and then concatenated,
                                      # 'joint' would mean on joint data vector
              }
    }

import hashlib
IDENT = hashlib.md5(f'{S}'.encode('utf-8')).hexdigest()
