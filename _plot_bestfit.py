import numpy as np
from matplotlib import pyplot as plt

from settings import S, IDENT
from compressed_data import CompressedData
from gpr import GPR
from _plot_style import *

run_hash = '9d56790a0f55a6885899ec32284b91bd'
assert IDENT == run_hash

with np.load(f'real_chain_{run_hash}.npz') as f :
    chain = f['chain'].reshape(-1, 2)
    lp = f['lp'].flatten()
idx = np.argmax(lp)
theta_bf = chain[idx]

dof = 12 + 2
chisq_red_bf = -2 * lp[idx] / dof
print(f'chisq_red_bf = {chisq_red_bf}')

cd = CompressedData()
real_data = cd.get_datavec('real')[0]
fid_data = cd.get_datavec('fiducial')
sigma = np.std(fid_data, axis=0)

emulator = GPR(cd)
theory_bf = emulator(theta_bf)

fig, ax = plt.subplots()

x = np.arange(len(real_data))
y = (real_data - theory_bf)/sigma
print(f'y ~ {np.min(y)} -- {np.max(y)}')

np.save(f'deltax_{run_hash}.npy', y)

ax.plot(x, y, linestyle='none', marker='o')
ax.axhline(0, color='grey', linestyle='dashed')

ax.set_xlabel('data vector index')
ax.set_ylabel('$(x_{\sf HSC} - x_{\sf bestfit})/\sigma$')

savefig(fig, 'bestfit')
