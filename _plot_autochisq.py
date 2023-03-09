import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.distributions import chi2

from data import Data
from loglike import LogLike
from _plot_style import *



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

savefig(fig, 'autochisq')
