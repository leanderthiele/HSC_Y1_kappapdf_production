""" Run one chain, can be imported by other files
We always sample S8, Om
"""

import sys
import os

import numpy as np
from scipy.optimize import minimize

import emcee

from loglike import LogLike
from settings import S, IDENT

np.seterr(all='raise') # not sure why we did this originally, keep...

def logprior (theta) :
    S8, Om = theta
    in_interval = lambda x, t: t[0] <= x <= t[1]

    if not in_interval(Om, S['prior']['Om']) :
        return -np.inf

    # we have uniform prior in sigma8
    if 's8' in S['prior'] :
        s8 = S8 / np.sqrt(Om/0.3)
        if not in_interval(s8, S['prior']['s8']) :
            return -np.inf
        return -0.5 * np.log(Om)

    # we have uniform prior in S_8
    if not in_interval(S8, S['prior']['S8']) :
        return -np.inf

    return 0


def logprob (theta, loglike) :
    try :
        lp = logprior(theta)
        if not np.isfinite(lp) :
            return -np.inf
        return lp + loglike(theta)
    except FloatingPointError :
        return -np.inf


def nll (theta, loglike) :
    # this is the optimization objective when finding the starting point
    # NOTE that in this case theta may be either [S8, Om] or [s8, Om]
    t = theta.copy()
    if 's8' in S['prior'] :
        # need to transform to S8
        t[0] *= np.sqrt(t[1]/0.3)
    return -logprob(t, loglike)


def get_init_theta (loglike) :
    ml_theta_start = np.array([0.8, 0.3])
    ml_sln = minimize(nll, ml_theta_start, args=loglike,
                      method='Powell',
                      bounds=[ S['prior']['S8' if 'S8' in S['prior'] else 's8'], S['prior']['Om'] ])
    if not ml_sln.success :
        print(f'*** Warning: minimization not successful. {ml_sln.message}', file=sys.stderr)
        ml_theta = ml_theta_start
    else :
        ml_theta = ml_sln.x

    rng = np.random.default_rng()
    init_theta = ml_theta + rng.normal(0, 1e-2, size=(S['mcmc']['nwalkers'], 2))

    # make sure these are all inside the prior (if not we replace by uniform samples)
    lo, hi = [ [p[ii] for p in S['prior'].values()] for ii in range(2) ]
    for ii in range(S['mcmc']['nwalkers']) :
        # use nll here because it is already in terms of the correct amplitude which makes it convenient
        if not np.isfinite(nll(init_theta[ii], loglike)) :
            init_theta[ii, :] = rng.uniform(lo, hi)
            
    # now transform back to S8 if necessary
    if 's8' in S['prior'] :
        init_theta[:, 0] *= np.sqrt(init_theta[:, 1]/0.3)
        ml_theta[0] *= np.sqrt(ml_theta[1]/0.3)

    return init_theta, ml_theta


def Sample (obs_case, obs_idx) :

    ll = LogLike(obs_case, obs_idx)
    init_theta, ml_theta = get_init_theta(ll)
    assert all(np.isfinite(logprior(t)) for t in init_theta)

    # increase the stretch parameter from its default (2) to decrease acceptance rate and correlation time
    sampler = emcee.EnsembleSampler(S['mcmc']['nwalkers'], 2, logprob, args=ll,
                                    moves=emcee.moves.StretchMove(a=5))
    sampler.run_mcmc(init_theta, S['mcmc']['nsteps'], progress=(obs_case=='real'))

    chain = sampler.get_chain(thin=S['mcmc']['thin'], discard=S['mcmc']['discard'])

    try :
        autocorr_times = sampler.get_autocorr_time(discard=S['mcmc']['discard'])
    except emcee.autocorr.AutocorrError :
        autocorr_times = np.full(2, float('nan'))

    acceptance_rates = sampler.acceptance_fraction

    print(f'autocorr times:\n{autocorr_times}\nacceptance rates:\n{acceptance_rates}')

    wrkdir = f'/scratch/gpfs/lthiele/HSC_Y1_chains/{obs_case}_{IDENT}'
    os.makedirs(wrkdir, exist_ok=True)
    fname = f'{wrkdir}/chain_{obs_idx}.npz'
    np.savez(fname,
             chain=chain, autocorr_times=autocorr_times, acceptance_rates=acceptance_rates,
             ml_theta=ml_theta, true_theta=ll.theta_real)
