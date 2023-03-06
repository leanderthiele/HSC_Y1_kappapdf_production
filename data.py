""" Utilities for data loading

General layout of data arrays is [..., zs, smooth, bin]
where ...  depends on the type of data
"""

import numpy as np

from settings import S

# fiducial point, order is important here!
FID = {
       'S8': 0.82 * np.sqrt(0.279 / 0.3),
       'Om': 0.279,
      }

# some data layout constants
ROOT = '/scratch/gpfs/lthiele/HSC_Y1_Nbody_sims'
NFID = 2268
NCOS = 100
NCOS_RND = 50
TNG_N_PER_POINT = 5
NSYST = {
         'stats_fiducial': NFID,
         'photoz/frankenz': 200,
         'photoz/mizuki': 200,
         'baryons/hydro': 1998//TNG_N_PER_POINT,
         'baryons/dark': 1998//TNG_N_PER_POINT,
         'nom': NFID,
         'real': 1,
        }
NSMOOTH_ALL = 8
NBINS_ALL = {'pdf': 19, 'ps': 14, }
NZS = len(S['zs'])

# the observational layout
FIELDS = {
          'wide12h': 14.20,
          'hectomap': 12.26,
          'vvds': 21.80,
          'xmm': 32.20,
          'gama09h': 32.80,
          'gama15h': 23.73,
         }
TOT_AREA = sum(FIELDS.values())

USE_STATS = list(filter(lambda s: s in S, ['pdf', 'ps', ]))


def transf (x, k) :
    """ possible non-linear transformation, execute only *after* cutting """
    if S[k]['rebin'] not in [None, 1, ] :
        x = x.reshape(*x.shape[:-1], -1, S[k]['rebin']).sum(axis=-1)

    if k == 'ps': 
        x *= 3e9
    elif S[k]['log'] :
        # avoid NaN, the chosen number is the minimum non-zero entry
        x = 13 + np.log(np.maximum(x, 1.2347965399050656e-06))
    else :
        x *= 3e1

    return x


def cut (x, k) :
    """ smoothing scale and bin cutting """
    delete_smooth = set(range(NSMOOTH_ALL)) - ( set(S[k]['smooth']) if k=='pdf' else {0, } )
    delete_bins = set(range(S[k]['high_cut'])) \
                  | set(range(NBINS_ALL[k] - S[k]['low_cut'], NBINS_ALL[k])) \
                  | {S[k]['delete'], } if k=='pdf' else set()
    x = np.delete(x, tuple(delete_smooth), axis=-2)
    x = np.delete(x, tuple(delete_bins), axis=-1)
    return x


def _get_fiducial_impl (k, systematic='stats_fiducial') :
    
    out = np.zeros((NSYST[systematic], NZS, NSMOOTH_ALL, NSMOOTH_ALL, NBINS_ALL[k]))

    for ii, zs_idx in enumerate(S['zs']) :
        
        # usual case (not kappaTNG)
        if not systematic or 'baryons' not in systematic :
            for field_name, field_area in FIELDS.items() :
                fname = f'{ROOT}/{systematic}/'\
                        f'{k if k=="pdf" else "power spectrum"}/{k if k=="pdf" else "clee"}_'\
                        f'{f"z{zs_idx}" if zs_idx else "singlez"}_all_smooths_{field_name}'\
                        f'{"" if k!="pdf" or not S[k]["unitstd"] or systematic=="real" else "_stdmap"}.npy'
                f = np.load(fname)
                if len(out) == 1 : # special case for reals :
                    f = f.reshape(1, -1)
                if k == 'pdf' :
                    # TODO read this again
                    out[:, ii, ...] += (field_area / TOT_AREA) \
                                       * (f / np.mean(np.sum(f, axis=-1), axis=0)[None, :, None])
                else :
                    out[:, ii, ...] += (field_area / total_area) * f

        # kappaTNG case
        else :
            
            for jj in range(len(out)) :
                for kk in range(TNG_N_PER_POINT) :
                    sim_idx = kk*nsims + jj # some non-consecutive thing to wash out potential correlations
                    fname = f'{ROOT}/{systematic}/'\
                            f'{k+("" if not S[k]["unitstd"] else "_unitstd") if k=="pdf" else "power_spectrum"}/'\
                            f'{k if k=="pdf" else "clee"}_{f"z{zs_idx}" if zs_idx else "singlez"}_'\
                            f'nsim_{sim_idx+1}.npy'
                    f = np.load(fname)
                    # TODO read this again
                    out[jj, ii, ...] += (1/TNG_N_PER_POINT) * (f / np.sum(f, axis=1)[:, None])

    # postprocessing
    out = cut(out, k)
    out = transf(out, k)
    return out.reshape(*out.shape[:1], -1)


def _get_cosmo_varied_impl (k, corrected=None) :

    out = np.zeros((NCOS, NCOS_RND, NZS, NSMOOTH_ALL, NBINS_ALL))

    for cc in range(NCOS) :
        for ii, zs_idx in enumerate(S['zs']) :
            for field_name, field_area in FIELDS.items() :
                fname = f'{ROOT}/stats_{f"{corrected}_" if corrected else ""}cosmo_varied/'\
                        f'{k if k=="pdf" else "power_spectrum"}/model{cc+1}/{k if k=="pdf" else "clee"}'\
                        f'_{f"z{zs_idx}" if zs_idx else "singlez"}_all_smooths_{field_name}_'\
                        f'{"" if k!="pdf" or not S[k]["unitstd"] else "_stdmap"}.npy'
                f = np.load(fname)
                if k == 'pdf' :
                    # TODO read this again
                    out[cc, :, ii, ...] += (field_area / TOT_AREA) \
                                           * (f / np.mean(np.sum(f, axis=-1), axis=0)[None, :, None])
                else :
                    out[cc, :, ii, ...] += (field_area / TOT_AREA) * f

    # postprocessing
    out = cut(out, k)
    out = transf(out, k)
    return out.reshape(*out.shape[:2], -1)


def _get_helper (fct, *args, **kwargs) :
    return np.concatenate([fct(k, *args, **kwargs) for k in USE_STATS], axis=-1)


def get_fiducial (systematic='stats_fiducial', cache={}) :
    """ returns [random, bin] """

    if systematic not in cache :
        cache[systematic] = _get_helper(_get_fiducial_impl, systematic=systematic)
    return cache[systematic].copy()


def get_cosmo_varied (corrected=None, cache={}) :
    """ returns [cosmology, random, bin] """

    if corrected not in cache :
        cache[corrected] = _get_helper(_get_cosmo_varied_impl, corrected=corrected)
    return cache[corrected].copy()


def get_cosmo_theta () :
    """ returns [N, 2] where theta=(S8, Om) """

    _, Om, s8 = np.loadtxt(f'{ROOT}/stats_cosmo_varied/omegam_sigma8_design3.dat',
                           unpack=True)
    S8 = s8 * np.sqrt(Om / 0.3)
    return np.stack([S8, Om, ], axis=-1)
