Code for PDF+power spectrum analysis of HSC Y1.

- `settings.py`: global settings, including a hash to identify runs
- `data.py`: load the data from disk
- `compressed_data.py`: implements MOPED compression
- `gpr.py`: Gaussian process emulator
- `gpr_optimization.py`: a helper script to find good settings
- `cov.py`: covariance matrix construction
- `sample.py`: run a Markov chain on some observation
- `trial_chains.py`: run chains on synthetic data
- `coverage_calcs.py`: utilities to produce the calibration diagnostics
- `real_chain.py`: run chain on real data
- `baryon_ratio.py`: reduce the kappaTNG data to a ratio
                    for contamination
- `_plot_*.py`: various plotting scripts
