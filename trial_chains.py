import sys
from sys import argv
import os

import numpy as np

from data import Data
from sample import Sample, logprior
from settings import S, IDENT

OBS_CASE = argv[1]
NCORES = int(argv[2])

WRKDIR = f'/scratch/gpfs/lthiele/hsc_chains/{OBS_CASE}_{IDENT}'


def get_possible_sims () :
    """ returns allowed simulation indices and their associated prior probabilities """

    data = Data()
    # for the fiducials, need to expand the leading unit axis
    theta_sims = data.get_cosmo(OBS_CASE).reshape(-1, theta_sims.shape[-1])

    # compute their logpriors
    logprior_sims = np.array([logprior(t) for t in theta_sims])

    nseeds = data.NSEEDS[OBS_CASE]
    obs_indices = np.arange(len(theta_sims) * nseeds, dtype=int)
    logprior_sims = logprior_sims.repeat(nseeds, axis=0)

    # select the ones that have non-zero prior
    select = np.isfinite(logprior_sims)
    obs_indices = obs_indices[select]
    logprior_sims = logprior_sims[select]

    # these are the real normalized probabilities
    prior_sims = np.exp(logprior_sims)
    prior_sims /= np.sum(prior_sims)

    return obs_indices, prior_sims


def one_chain (idx) :
    
    fname = f'{WRKDIR}/chain_{idx}.npz'

    try :
        result = Sample(OBS_CASE, idx)
    except Exception as e :
        print(f'***Failed for idx={idx}: {e}', file=sys.stderr)
        return

    np.savez(fname, **result)


if __name__ == '__main__' :
    
    os.makedirs(WRKDIR, exist_ok=True)
    rng = np.random.default_rng() # start with some seed

    obs_indices, prior_sims = get_possible_sims()

    # this is really just a random re-ordering (with preference for the high-prior sims to come first)
    # if the prior is the same for all, everything is ok
    # otherwise, it really only makes sense if we use a fraction of these
    # note that we make use of the fact that the samples have already been filtered to discard zero prior
    obs_indices = rng.choice(obs_indices, size=len(obs_indices), replace=False, p=prior_sims)

    with mp.Pool(NCORES) as pool :
        pool.map(one_chain, obs_indices)
