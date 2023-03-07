import numpy as np

from settings import S

class Data :

    # where the files are
    ROOT = '/scratch/gpfs/lthiele/HSC_Y1_Nbody_sims'

    # fiducial point
    FID = {
           'S8': 0.82 * np.sqrt(0.279 / 0.3),
           'Om': 0.279,
          }
    FID_THETA = np.array(list(FID.values()))

    # data layout
    NCOS = 100
    # TODO TNG not implemented yet, probably do this through ratio and get into same format
    # TNG_N_PER_POINT = 5
    NSEEDS = {
              'cosmo_varied': 50,
              'fiducial': 2268,
              'photoz/frankenz': 210,
              'photoz/mizuki':   210,
              'mbias/mbias_minus': 100,
              'mbias/mbias_plus':  100,
    # TODO we don't have these available yet
    #         'baryons/hydro': 1998//TNG_N_PER_POINT,
    #         'baryons/dark':  1998//TNG_N_PER_POINT,
    # TODO this one vanished, not sure if we ever needed it
    #         'nom': NFID,
              'real': 1,
             }
    NSMOOTH_ALL = 8
    NBINS_ALL = {'pdf': 19, 'ps': 14, }

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

    # some small computations
    USE_STATS = list(filter(lambda s: s in S, ['pdf', 'ps', ]))
    NZS = { stat: len(S[stat]['zs']) for stat in USE_STATS }
    DELETE_SMOOTH = {
                     stat: set(range(NSMOOTH_ALL)) - ( set(S[stat]['smooth']) if stat=='pdf' else {0, } )
                     for stat in USE_STATS
                    }
    DELETE_BINS = {
                   stat: set(range(S[stat]['high_cut'])) \
                         | set(range(NBINS_ALL[stat] - S[stat]['low_cut'], NBINS_ALL[stat])) \
                         | {S[stat]['delete'], } if stat=='pdf' else set()
                   for stat in USE_STATS
                  }
    NDIMS = {
             stat: NZS[stat] * (NSMOOTH_ALL - len(DELETE_SMOOTH[stat])) * (NBINS_ALL - len(DELETE_BINS[stat]))
             for stat in USE_STATS
            }

    # avoid repeated disk-IO by having evaluated results in this cache
    _CACHE = {}

    def __init__ (self) :
        pass
    
    
    def get_datavec (self, case) :
        """ main public method of this class """
        return np.concatenate([self._get_data_array(stat, case) for stat in Data.USE_STATS],
                              axis=-1)

    
    def get_cosmo (self, case) :
        """ get cosmology theta vector
        if case==cosmo_varied, [N, 2, ], else [2, ],
        in the order S8, Om
        """
        if case == 'cosmo_varied' :
            _, Om, s8 = np.loadtxt(f'{Data.ROOT}/stats_cosmo_varied/omegam_sigma8_design3.dat',
                                   unpack=True)
            S8 = s8 * np.sqrt(Om / 0.3)
            return np.stack([S8, Om, ], axis=-1)
        else :
            return Data.FID_THETA.copy()


    def get_used_stats (self) :
        return self.USE_STATS.copy()


    def get_stat_mask (self, stat) :
        """ return boolean mask that is True where this stat lives """
        return np.concatenate([np.full(Data.NDIMS[s], s==stat, dtype=bool) for s in Data.USE_STATS])


    def _get_data_array (self, *args) :
        """ wrapper around _stack_data which implements caching """
        key = '_'.join(args)
        if key not in Data._CACHE :
            Data._CACHE[key] = self._stack_data(*args)
        return Data._CACHE[key].copy() # for some extra safety, probably not required but really cannot hurt


    def _stack_data (self, stat, case) :
        """ load all files into memory and stack them into a nice array, includes cuts and transformations
        output = [ (cosmo_idx), seed, data bin ]
        """
        if case == 'cosmo_varied' :
            out = np.stack([self._stack_data_helper(stat, case, model=cc+1) for cc in range(Data.NCOS)], axis=0)
        else :
            out = self._stack_data_helper(stat, case)
        # cuts
        out = np.delete(out, tuple(Data.DELETE_SMOOTH[stat]), axis=-2)
        out = np.delete(out, tuple(Data.DELETE_BINS[stat]), axis=-1)
        # transformations
        if (r := S[stat]['rebin']) not in [None, 1, ] :
            out = out.reshape(*out.shape[:-1], -1, r).sum(axis=-1)
        if stat == 'ps' :
            out *= 3e9
        elif S[stat]['log'] :
            out = 13 + np.log(np.maximum(out, 1.2347965399050656e-06))
        else : # pdf without log
            out *= 3e1
        return out.reshape(*out.shape[:(2 if case=='cosmo_varied' else 1)], -1)


    def _stack_data_helper (self, stat, case, model=None) :
        """ helper for _stack_data, working with a single model (for fiducials that is the only one)
        output = [seed, zs, smooth, nbins]
        does not perform cutting
        """
        out = np.zeros((Data.NSEEDS[case], Data.NZS[stat], Data.NSMOOTH_ALL, Data.NBINS_ALL[stat]))
        for ii, zs_idx in enumerate(S[stat]['zs']) :
            for field, area in Data.FIELDS.items() :
                f = np.load(self._fname(stat, case, zs_idx, field, model))
                if len(out) == 1 : # special case for real
                    f = f.reshape(1, *f.shape)
                if stat == 'pdf' : # normalize TODO read this again
                    f = f / np.mean(np.sum(f, axis=-1), axis=0)[None, :, None]
                out[:, ii, ...] += (area / Data.TOT_AREA) * f
        return out


    def _fname (self, stat, case, zs_idx, field, model=None) :
        """ return the file name 
        stat ... one of [pdf, ps, ]
        case ... one of the keys in NSEEDS
        zs_idx ... 0 for singlez, 1-4 otherwise
        field ... one of the keys in FIELDS
        model ... if case==cosmo_varied, the cosmology model index
        """
        assert stat in Data.USE_STATS 
        assert case in Data.NSEEDS.keys()
        assert zs_idx in [0, 1, 2, 3, 4, ]
        assert field in Data.FIELDS.keys()
        out = f'{Data.ROOT}/stats_{case}'
        out = f'{out}/{stat if stat=="pdf" else "power_spectrum"}'
        if case == 'cosmo_varied' :
            assert model is not None
            assert 1 <= model <= Data.NCOS
            out = f'{out}/model{model}'
        out = f'{out}/{stat if stat=="pdf" else "clee"}'
        out = f'{out}_{f"z{zs_idx}" if zs_idx else "singlez"}'
        out = f'{out}_all_smooths_{field}'
        if stat == 'pdf' and S[stat]['unitstd'] and case != 'real' :
            out = f'{out}_stdmap'
        out = f'{out}.npy'
        return out



class DataWrapper :
    """ other classes implementing the same interface as Data can subclass this
    It checks if a method has been overwritten and if not uses the original one
    """

    def __init__ (self, data) :
        self.data = data

    def __getattr__ (self, name) :
        # NOTE that getattr is only called after other paths have been exhausted
        assert name.startswith('get')
        return getattr(self.data, name)
