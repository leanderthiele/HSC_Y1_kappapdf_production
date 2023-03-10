from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.gaussian_process import kernels

from data import Data
from gpr import GPR

# here we test
kernel = kernels.RBF(length_scale=1, length_scale_bounds='fixed')

data = Data()

# for simple scoring this should be enough
cov = np.cov(data.get_datavec('fiducial'), rowvar=False) / data.get_nseeds('cosmo_varied')
covinv = np.linalg.inv(cov)

# our test data
theta_sims = data.get_cosmo('cosmo_varied')
y_sims = np.mean(data.get_datavec('cosmo_varied'), axis=1)

# collect data
chisq = []
for ii in tqdm(range(len(theta_sims))) :
    emulator = GPR(data, test_idx=ii, kernel=kernel)
    y_pred = emulator(theta_sims[ii])
    delta = y_pred - y_sims[ii]
    chisq.append(np.einsum('a,ab,b->', delta, covinv, delta)/y_sims.shape[-1])
chisq = np.array(chisq)

print(chisq)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.scatter(*theta_sims.T, c=chisq)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('$\chi^2$')
ax.set_xlabel('$S_8$')
ax.set_ylabel('$\Omega_m$')
ax.set_title(f'{kernel}')

fig.savefig('gpr_optimization.png')
