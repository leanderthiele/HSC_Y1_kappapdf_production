import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

from data import get_fiducial, get_cosmo_varied, get_cosmo_theta, FID


def _construct_gpr_input (xcosmo) :
    return xcosmo - np.array(FID.values())[None, :]


class GPRInterpolator :

    def __init__ (self, gpr, y_avg) :
        self.gpr = gpr
        self.y_avg = y_avg

    def __call__ (self, x) :
        ypred = self.gpr.predict(_construct_gpr_input(x.reshape(-1, 2)))
        if self.y_avg is not None :
            ypred = (1 + ypred) * self.y_avg
        return ypred


def get_gpr (test_idx=None, reduction=np.mean, A=None, x=None) :
    """ main routine of this module
    
    test_idx  ... exclude this cosmology from training set (for performance testing)
    reduction ... how the random seeds are collapsed. Usually we use mean for the theory
                  predicted signal, but can use other reductions (used in cov)
    A         ... matrix used to linearly transform the data
    x         ... do the gpr for a completely different target, needs to be of shape
                  [NCOS, NDIMS, ]
                  (in which case neither reduction nor A are being used)
    """

    theta = get_cosmo_theta()
    if x is None :
        yfid = get_fiducial()
        y = get_cosmo_varied()
        if A is not None :
            yfid = np.einsum('ab,ib->ia', A, yfid)
            y = np.einsum('ab,ijb->ija', A, y)
        y_avg = reduction(yfid, axis=0)
    else :
        y = x
        y_avg = None

    if test_idx is not None :
        theta = np.delete(theta, test_idx, axis=0)
        y = np.delete(y, test_idx, axis=0)

    if x is None :
        y = reduction(y, axis=1)
        y = y / y_avg[None, ...] - 1

    theta_norm = _construct_gpr_input(theta)
    gpr = GPR(kernel=kernels.RBF(length_scale=1, length_scale_bounds='fixed'))\
            .fit(theta_norm, y)

    return GPRInterpolator(gpr, y_avg)
