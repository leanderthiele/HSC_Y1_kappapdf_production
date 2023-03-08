""" GLOBAL SETTINGS

smoothing scales are
0 --  1 arcmin
1 --  2 arcmin
2 --  5 arcmin
3 --  7 arcmin
4 --  8 arcmin
5 -- 10 arcmin
6 -- 15 arcmin
7 -- 25 arcmin

The cov_mode setting works as follows:
fixed     ... block is kept fixed at the fiducial point
gpr       ... block's inverse is emulated
gpr_scale ... same as gpr but scale s.t. agrees at fiducial point
scale     ... block's corr is kept fixed at fiducial point
              but scaling with sigma (from emulator) is performed
In all cases, we keep the cross-correlations between different
summary statistics fixed at their values at the fiducial point.

The deriv_mode in moped has two choices
gpr     ... finite differencing using a Gaussian process emulator
lstsq_N ... linear regression using N cosmo-varied points surrounding the fiducial point
In tests, it appears that gpr output is very stable under changing the step size,
wheres lstsq tends to disagree with gpr and also is more unstable under changing the number of points N
(qualitatively gpr and lstsq are in agreement though).
lstsq approaches the gpr result as N is increased to relatively large (~20)
"""


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
     #'ps': {
     #       'rebin': 1,
     #       'zs': [0, ],
     #       'low_cut': 4,
     #       'high_cut': 6,
     #       'cov_mode': 'scale',
     #      },

     # this is a bit ugly but who cares (special case when MOPED in joint mode)
     # probably never used...
     'joint': { 'cov_mode': 'gpr', },
     
     # moped settings
     'moped': {
               'deriv_mode': 'gpr', # either lstsq_N or gpr
               'apply_to': ['pdf', ], # list means separate and then concatenated,
                                      # 'joint' would mean on joint data vector
              },
     
     # sampling settings, autocorrelation time around 20 so it is fine
     'mcmc': {
              'nwalkers': 128,
              'nsteps': 2000,
              'discard': 200,
              'thin': 20,
             },

     # can switch s8 and S8 here to change the prior
     # ordering is important here!!!
     'prior': {
               'S8': (0.45, 1.05),
               # 's8': (0.60, 1.10),
               'Om': (0.20, 0.40),
              },
    }

import hashlib
IDENT = hashlib.md5(f'{S}'.encode('utf-8')).hexdigest()
