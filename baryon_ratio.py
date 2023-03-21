import os
import os.path
from glob import glob

import numpy as np

from settings import SIM_ROOT

for stat in ['pdf_stdmap', 'pdf', 'ps', ] :
    
    print(f'stat = {stat}')
    
    for zstr in ['singlez', 'z1', 'z2', 'z3', 'z4', ] :

        print(f'\tz = {zstr}')

        pattern = f'{"pdf" if stat.startswith("pdf") else "power_spectrum"}'
        pattern = f'{pattern}/{"pdf" if stat.startswith("pdf") else "clee"}_{zstr}'
        outfile = pattern
        pattern = f'{pattern}_nsim_[0-9]*'
        if 'stdmap' in stat :
            pattern = f'{pattern}_stdmap'
            outfile = f'{outfile}_stdmap'
        pattern = f'{pattern}.npy'
        outfile = f'{SIM_ROOT}/ratio_baryon/{outfile}.npy'

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        tng_dark_fnames = glob(f'{SIM_ROOT}/kappaTNG/dark/{pattern}')
        print(f'Found {len(tng_dark_fnames)} dark files ')
        tng_hydro_fnames = glob(f'{SIM_ROOT}/kappaTNG/hydro/{pattern}')
        print(f'Found {len(tng_hydro_fnames)} hydro files')
        assert len(tng_hydro_fnames) == len(tng_dark_fnames)

        tng_dark_data = np.stack([np.load(fname) for fname in tng_dark_fnames])
        tng_hydro_data = np.stack([np.load(fname) for fname in tng_hydro_fnames])

        ratio = np.mean(tng_hydro_data, axis=0) / np.mean(tng_dark_data, axis=0)

        np.save(outfile, ratio)
