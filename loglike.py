import numpy as np

from data import Data
from compressed_data import CompressedData
from cov import Cov
from gpr import GPR
from settings import S


class LogLike :

    # to avoid repeatedly refitting the GPRs during testing
    _CACHE = {}
    

    def __init__ (self, obs_case, obs_idx) :
        """ constructor
        obs_case ... one of the entries in Data.NSEEDS
                     (can have ~ratio) added
        obs_idx  ... the observation index in the flattened array, 
                     needs to be zero for real data
        """

        self.data = CompressedData()

        if obs_case.startswith('cosmo_varied') :
            test_idx = obs_idx // self.data.get_nseeds(obs_case)
            self.theta_real = self.data.get_cosmo(obs_case)[test_idx]
        else :
            test_idx = None
            self.theta_real = self.data.get_cosmo(obs_case)

        datavecs = self.data.get_datavec(obs_case)

        self.observation = datavecs.reshape(-1, datavecs.shape[-1])[obs_idx]
        self.cov_obj = self._get_cov(test_idx)
        self.gpr = self._get_gpr(test_idx)

    
    def __call__ (self, theta) :
        
        model = self.gpr(theta)
        delta = self.observation - model
        covinv = self.cov_obj.covinv(theta)
        chisq = np.einsum('a,ab,b->', delta, covinv, delta)
        return -0.5 * chisq


    def _get_cov (self, test_idx) :
        key = f'cov_{test_idx}'
        if key not in LogLike._CACHE :
            LogLike._CACHE[key] = Cov(self.data, test_idx=test_idx)
        return LogLike._CACHE[key]


    def _get_gpr (self, test_idx) :
        key = f'gpr_{test_idx}'
        if key not in LogLike._CACHE :
            subsample = None if 'mean_emulator_subsample' not in S \
                        else S['mean_emulator_subsample']
            LogLike._CACHE[key] = GPR(self.data, test_idx=test_idx, subsample=subsample)
        return LogLike._CACHE[key]

