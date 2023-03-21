import os
import os.path

import numpy as np

from settings import SIM_ROOT

smooth_scales = [1, 2, 5, 7, 8, 10, 15, 25, ]

def get_ratio (stat, zstr, IA_ampl, theta, method) :
    # output shape is [bins]
    fid_fname = f'{SIM_ROOT}/stats_IA/{"pdf" if stat.startswith("pdf") else "clee"}_{zstr}_0_theta_{theta}.npy'
    if not os.path.isfile(fid_fname) :
        # we don't have all smoothing scales available for IA
        return np.full(19 if stat.startswith('pdf') else 14, float('nan'))
    var_fname = f'{SIM_ROOT}/stats_IA/{"pdf" if stat.startswith("pdf") else "clee"}_{zstr}_{IA_ampl}_theta_{theta}.npy'
    x0 = np.load(fid_fname)
    x1 = np.load(var_fname)
    if method == 'simple' :
        return np.mean(x1, axis=0) / np.mean(x0, axis=0)
    else :
        raise NotImplementedError()

for stat in ['pdf', 'ps', ] :
    
    for zstr in ['singlez', 'z1', 'z2', 'z3', 'z4', ] :
        
        for IA_ampl in ['032', '118', ] :
            
            for method in ['simple', ] :
                
                ratio = np.stack([get_ratio(stat, zstr, IA_ampl, theta, method) for theta in smooth_scales],
                                 axis=0)
                outfile = f'{SIM_ROOT}/ratio_IA/{IA_ampl}_{method}/'
                outfile = f'{outfile}/{"pdf" if stat.startswith("pdf") else "power_spectrum"}'
                outfile = f'{outfile}/{"pdf" if stat.startswith("pdf") else "clee"}_{zstr}'
                if stat == 'pdf' :
                    # we only have the stdmap here from Gabriela
                    outfile = f'{outfile}_stdmap'
                outfile = f'{outfile}.npy'

                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                
                np.save(outfile, ratio)
