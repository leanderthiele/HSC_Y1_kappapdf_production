import sys
from sys import argv
import os
import traceback

import multiprocessing as mp

import numpy as np

from data import Data
from sample import Sample, logprior
from coverage_calcs import Oneminusalpha, Ranks
from settings import S, IDENT

OBS_CASE = argv[1]
NCORES = int(argv[2])

WRKDIR = f'/scratch/gpfs/lthiele/hsc_chains/{OBS_CASE.replace("~", "-")}_{IDENT}'


def get_possible_sims () :
    """ returns allowed simulation indices and their associated prior probabilities """

    data = Data()
    # for the fiducials, need to expand the leading unit axis
    theta_sims = data.get_cosmo(OBS_CASE)
    theta_sims = theta_sims.reshape(-1, theta_sims.shape[-1])

    # compute their logpriors
    logprior_sims = np.array([logprior(t) for t in theta_sims])

    nseeds = data.get_nseeds(OBS_CASE)
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


class Workers :

    def __init__ (self, summary_fname) :
        self.summary_fname = summary_fname


    def __call__ (self, idx) :

        
        try :
            result = Sample(OBS_CASE, idx)
        except Exception as e :
            print(f'***Sample failed for idx={idx}: {traceback.format_exc()}',
                  file=sys.stderr)
            return

        chain_fname = f'{WRKDIR}/chain_{idx}.npz'
        np.savez(chain_fname, **result)

        chain = result['chain']
        chain = chain.reshape(-1, chain.shape[-1])
        mean = np.mean(chain[:, 0])
        std = np.std(chain[:, 0])
        line = f'{mean:.8f} {std:.8f}'

        if OBS_CASE.startswith('cosmo_varied') :
            true_theta = result['true_theta']
            try :
                oma, xmap = Oneminusalpha(chain[:, 0], true_theta[0])
            except Exception as e :
                print(f'***Oneminusalpha failed for idx={idx}: {traceback.format_exc()}',
                      file=sys.stderr)
                oma = -1
                xmap = float('nan')

            try :
                ranks = Ranks(chain, true_theta)
            except Exception as e :
                print(f'***Ranks failed for idx={idx}: {traceback.format_exc()}',
                      file=sys.stderr)
                ranks = np.full(chain.shape[-1], -1)
            ranks_str = ' '.join(map(lambda s: f'{s:.8f}', ranks))
            line = f'{oma:.8f} {ranks_str} {line} {xmap}'
        
        # make sure we don't mess up the output
        lock.acquire()
        try :
            with open(self.summary_fname, 'a') as f :
                f.write(f'{idx:5} {line}\n')
        except Exception as e :
            print(f'***Writing line to file failed for idx={idx}: {traceback.format_exc()}',
                  file=sys.stderr)
        finally :
            lock.release()


    @staticmethod
    def init_pool_process (the_lock) :
        global lock
        lock = the_lock



if __name__ == '__main__' :
    
    os.makedirs(WRKDIR, exist_ok=True)
    rng = np.random.default_rng() # start with some seed

    obs_indices, prior_sims = get_possible_sims()

    # this is really just a random re-ordering (with preference for the high-prior sims to come first)
    # if the prior is the same for all, everything is ok
    # otherwise, it really only makes sense if we use a fraction of these
    # note that we make use of the fact that the samples have already been filtered to discard zero prior
    obs_indices = rng.choice(obs_indices, size=len(obs_indices), replace=False, p=prior_sims)

    # for testing
    if OBS_CASE.startswith('cosmo_varied') and 'mean_emulator_subsample' in S :
        print('*** Doing the test in which we test on different augmentations than trained for!',
              file=sys.stderr)
        subsample = S['mean_emulator_subsample']
        nseeds = Data().get_nseeds(OBS_CASE)
        used_seeds = np.random.default_rng(subsample)\
                         .choice(nseeds, size=subsample, replace=False)
        obs_indices = list(filter(lambda idx: (idx % nseeds) not in used_seeds, obs_indices))


    # some random number so different slurm jobs don't interfere
    rnd = rng.integers(2**63)

    info_fname = f'{WRKDIR}/settings_{rnd}.info'
    with open(info_fname, 'w') as f :
        f.write(f'{S}\n')

    if OBS_CASE.startswith('cosmo_varied') :
        # this is for getting calibration checks
        summary_fname = f'{WRKDIR}/coverage_data_{rnd}.dat'
        with open(summary_fname, 'w') as f :
            f.write('# index, oneminusalpha, ranks..., mean, std, map(S8)\n')
    else :
        # this is for getting bias checks
        summary_fname = f'{WRKDIR}/bias_data_{rnd}.dat'
        with open(summary_fname, 'w') as f :
            f.write('# index, mean, std\n')

    LOCK = mp.Lock()
    workers = Workers(summary_fname)

    with mp.Pool(NCORES, initializer=Workers.init_pool_process, initargs=(LOCK, )) as pool :
        pool.map(workers, obs_indices)
