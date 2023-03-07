import numpy as np

from tqdm import tqdm

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
        obs_idx  ... the observation index in the flattened array, 
                     needs to be zero for real data
        """

        self.data = CompressedData()

        if obs_case == 'cosmo_varied' :
            test_idx = obs_idx // Data.NSEEDS[obs_case]
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
            LogLike._CACHE[key] = GPR(self.data, test_idx=test_idx)
        return LogLike._CACHE[key]


# TESTING
if __name__ == '__main__' :
    from matplotlib import pyplot as plt
    from scipy.stats.distributions import chi2
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def chisq (ll) :
        # returns chisq w.r.t. itself
        return -2 * ll(ll.theta_real)

    # first get chi-squared values at fiducial point
    fid_chisq = []
    for ii in tqdm(range(Data.NSEEDS['fiducial'])) :
        ll = LogLike('fiducial', ii)
        fid_chisq.append(chisq(ll))
    fid_chisq = np.array(fid_chisq)

    # collect garbage
    LogLike._CACHE.clear()

    # now at cosmo_varied points
    cosmo_varied_chisq, theta = [], []

    for ii in tqdm(range(Data.NCOS)) :
        this_cosmo_chisq = []
        for jj in range(Data.NSEEDS['cosmo_varied']) :
            ll = LogLike('cosmo_varied', ii*Data.NSEEDS['cosmo_varied']+jj)
            this_cosmo_chisq.append(chisq(ll))
        cosmo_varied_chisq.append(this_cosmo_chisq)
        theta.append(ll.theta_real)

        # collect garbage
        LogLike._CACHE.clear()

    cosmo_varied_chisq = np.array(cosmo_varied_chisq)
    theta = np.array(theta) # [N, 2]

    fig, ax = plt.subplots(nrows=3, figsize=(7, 14))
    dof = len(ll.observation) # TODO do we have to subtract 1 here?

    for ii, (a, chisq, label) in enumerate(zip(ax[:-1],
                                               [fid_chisq, cosmo_varied_chisq.flatten(), ],
                                               ['fiducial', 'cosmo_varied', ])) :
        _, e, _ = a.hist(chisq, bins=50, density=True, histtype='step', range=(0, 5*dof),
                         label=f'simulations {label}')
        dist = chi2(dof)
        expected = (dist.cdf(e[1:]) - dist.cdf(e[:-1])) / (e[1:]-e[:-1])
        c = 0.5 * (e[1:] + e[:-1])
        a.plot(c, expected, label='$\chi^2$ pdf')

        a.set_xlabel('$\chi^2$')
        a.set_ylabel('$P(\chi^2)$')
        a.legend(loc='upper right')

    S8, Om = theta.T
    im = ax[-1].scatter(Om, S8, c=np.mean(cosmo_varied_chisq, axis=-1)/dof, norm=LogNorm())
    divider = make_axes_locatable(ax[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'$\langle \chi_{\sf red}^2 \rangle$')
    ax[-1].set_xlabel('$\Omega_m$')
    ax[-1].set_ylabel('$S_8$')
    
    fig.savefig('loglike_chisq_check.pdf', bbox_inches='tight')
