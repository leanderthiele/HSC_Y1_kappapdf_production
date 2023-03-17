import sys

import numpy as np

from scipy.optimize import root_scalar, basinhopping
from sklearn.neighbors import KernelDensity

# LFT: copied this pretty much unchanged from nuvoid project

def Ranks (chain, true_theta) :
    """ this returns an array [Ntheta, ], normalized to [0, 1]
    make sure the chain is flattened!
    """
    ranks = np.sum(true_theta[None, :] > chain, axis=0, dtype=int)
    return ranks.astype(float) / len(chain)


def Oneminusalpha (samples, true_theta) :
    """ computes the quantity required for coverage test
    remember that this is a 1-D test, so samples is of shape [N, ],
    true_theta is a scalar
    """

    std = np.std(samples)
    
    # approximate continuous posterior
    # choice of bandwidth is visual heuristic, but should hopefully not matter too much
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01*std).fit(samples.reshape(-1, 1))
    nll = lambda x : -kde.score_samples(np.array([[x]]).reshape(-1,1))

    prior = (np.min(samples), np.max(samples))

    # get some feel for the posterior
    edges = np.linspace(*prior, num=501)
    centers = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(samples, bins=edges)
    x0 = centers[np.argmax(h)]
    xlo = max((prior[0], x0-std))
    xhi = min((prior[1], x0+std))
    while not np.isfinite(nll(xlo)) :
        xlo += 1e-2 * std
    while not np.isfinite(nll(xhi)) :
        xhi -= 1e-2 * std

    # find maximum posterior so we know in which direction to go
    try :
        sln = basinhopping(nll, x0, T=0.1, niter=10, minimizer_kwargs={'bounds': [(xlo, xhi), ]})
    except ValueError : # rare case
        print(f'*** basinhopping failed', file=sys.stderr)
        return -1, float('nan')

    xmap = sln.x.item()
    side = 'l' if true_theta<xmap else 'r'
    ytarg = nll(true_theta)
    edge = prior[1 if side=='l' else 0]
    if nll(edge) < ytarg :
        # pathological one-sided case
        xsln = edge
    else :
        ftarg = lambda x : ytarg - nll(x)
        bounds = (xmap, np.max(samples)) if side=='l' else (np.min(samples), xmap)
        try :
            sln = root_scalar(ftarg, bracket=bounds)
            xsln = sln.root
        except Exception as e :
            if str(e) == 'f(a) and f(b) must have different signs' :
                # extremely rare case in which true_theta is exactly at the maximum posterior
                return 0.0, xmap
            print(e, file=sys.stderr)
            return -1, xmap

    xmin, xmax = (true_theta, xsln) if side=='l' else (xsln, true_theta)
    return np.sum((samples>xmin)*(samples<xmax)) / len(samples), xmap
