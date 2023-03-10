from glob import glob
import re

import numpy as np

from data import Data

ROOT = '/scratch/gpfs/lthiele/hsc_chains'

# only consider runs with these true values
restrict = {
            'S8': (0.75, 0.85),
            'Om': (0.25, 0.35),
           }

def GetTestTrials (run_hashes, N=None, obs_case='cosmo_varied') :
    """ return N random trial indices that fall into restrict and where we have chains in all runs """

    data = Data()
    
    if obs_case.startswith('cosmo_varied') :
        theta_sims = data.get_cosmo('cosmo_varied')
        allowed_cosmo_indices = []
        in_interval = lambda x, t: t[0] <= x <= t[1]
        for ii, theta in enumerate(theta_sims) :
            if all(in_interval(t, r) for t, r in zip(theta, restrict.values())) :
                allowed_cosmo_indices.append(ii)

    # get the runs that we have available
    pattern = re.compile('chain_([0-9]*).npz')
    all_avail_indices = []
    all_avail_cosmo_indices = []
    for run_hash in run_hashes :
        chain_fnames = glob(f'{ROOT}/{obs_case.replace("~", "-")}_{run_hash}/chain_[0-9]*.npz')
        avail_indices = [int(pattern.search(chain_fname)[1]) for chain_fname in chain_fnames]
        avail_cosmo_indices = [idx//data.get_nseeds(obs_case) for idx in avail_indices]
        if obs_case.startswith('cosmo_varied') :
            # filter by our restriction
            avail_indices, avail_cosmo_indices = zip(*((idx, cosmo_idx) \
                                                       for idx, cosmo_idx in zip(avail_indices, avail_cosmo_indices) \
                                                       if cosmo_idx in allowed_cosmo_indices))
        all_avail_indices.append(avail_indices)
        all_avail_cosmo_indices.append(avail_cosmo_indices)

    # first try to get exactly the same runs, maybe we have enough
    # if we have about a thougsand runs this is usually enough
    avail_indices = list(set.intersection(*map(set, all_avail_indices)))

    if N is not None :
        assert len(avail_indices) >= N
        avail_indices = np.random.default_rng().choice(avail_indices, size=N, replace=False)

    return avail_indices
