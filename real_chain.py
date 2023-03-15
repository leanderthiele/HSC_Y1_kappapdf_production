import numpy as np

from sample import Sample
from settings import S, IDENT

result = Sample('real', 0)
np.savez(f'real_chain_{IDENT}.npz', **result)
