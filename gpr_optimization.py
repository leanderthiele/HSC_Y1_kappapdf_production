from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.gaussian_process import kernels

from data import Data
from gpr import GPR

# here we test
kernel = kernels.RBF(length_scale=3, length_scale_bounds='fixed')
# kernel = kernels.RationalQuadratic(length_scale=3, alpha=2.5, length_scale_bounds='fixed', alpha_bounds='fixed')

data = Data()

# for simple scoring this should be enough
cov = np.cov(data.get_datavec('fiducial'), rowvar=False) / data.get_nseeds('cosmo_varied')
covinv = np.linalg.inv(cov)

# our test data
theta_sims = data.get_cosmo('cosmo_varied')
y_sims = np.mean(data.get_datavec('cosmo_varied'), axis=1)

# collect data
chisq = []
sgn_chisq = []
for ii in tqdm(range(len(theta_sims))) :
    emulator = GPR(data, test_idx=ii, kernel=kernel)
    y_pred = emulator(theta_sims[ii])
    delta = y_pred - y_sims[ii]
    chisq.append(np.einsum('a,ab,b->', delta, covinv, delta)/y_sims.shape[-1])
    sgn_chisq.append(np.sum(delta * np.sqrt(np.diagonal(covinv))))
chisq = np.array(chisq)
sgn_chisq = np.array(sgn_chisq)
print(np.median(chisq))

print(chisq)

fig, ax = plt.subplots(figsize=(7, 7))
vmax = np.quantile(np.fabs(sgn_chisq), 0.9)
im = ax.scatter(*theta_sims.T, c=sgn_chisq, vmin=-vmax, vmax=vmax, cmap='seismic')
for t, c in zip(theta_sims, chisq) :
    ax.text(*t, f'{c:.1f}', va='bottom', ha='center', fontsize='xx-small', transform=ax.transData)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax.set_xlabel('$S_8$')
ax.set_ylabel('$\Omega_m$')
ax.set_title(f'{kernel}')

fig.savefig('gpr_optimization.png')
