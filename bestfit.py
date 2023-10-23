import numpy as np

runs = {
        '22927c646f516731cb39e41daab9e6e5': '$C_\ell^{\kappa\kappa}$',
        'fd47089b3f34889e50653bbb4ebeff98': 'PDF', # 5, 7, 10 arcmin
        '9d56790a0f55a6885899ec32284b91bd': 'PDF+$C_\ell^{\kappa\kappa}$', # 5, 7, 10 arcmin
       }

for run_hash, run_info in runs.items() :
    
    with np.load(f'real_chain_{run_hash}.npz') as f :
        chain = f['chain']
        lp = f['lp']

    chain = chain.reshape(-1, chain.shape[-1])
    lp = lp.flatten()

    idx = np.argmax(lp)

    dof = 0
    if 'PDF' in run_info :
        dof += 2
    if 'ell' in run_info :
        dof += 12

    chisq_red = -2 * lp[idx] / dof
    print(f'{run_info}: dof = {dof}, chisq_red_bf = {chisq_red}, chisq_bf = {-2 * lp[idx]}')
