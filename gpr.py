import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor, kernels


class GPR :
    

    def __init__ (self, data, reduction=np.mean, test_idx=None) :
        """ constructor
        data ... something that behaves like a Data instance (but can be something different, e.g. compressed)
                 Needs to implement get_cosmo and get_datavec methods
        reduction ... the function used to collapse over random seeds
        test_idx ... if given, remove this cosmology from the training set (for testing)
        """

        # TODO I think there are cases where we don't do the normalization by yfid
        #      (because it's not possible or something)

        self.xfid = data.get_cosmo('fiducial')
        x = self._norm_x(data.get_cosmo('cosmo_varied'))

        self.yfid = reduction(data.get_datavec('fiducial'), axis=0)
        y = reduction(data.get_datavec('cosmo_varied'), axis=1) / self.yfid[None, ...] - 1

        if test_idx is not None :
            x = np.delete(x, test_idx, axis=0)
            y = np.delete(y, test_idx, axis=0)

        self.gpr = GaussianProcessRegressor(kernel=kernels.RBF(length_scale=1, length_scale_bounds='fixed'))\
                        .fit(x, y)


    def __call__ (self, x) :
        y = self.gpr.predict(self._norm_x(x.reshape(-1, 2)))
        if self.yfid is not None :
            y = (1 + y) * self.yfid
        return y


    def _norm_x (self, x) :
        return x - self.xfid[None, :]
