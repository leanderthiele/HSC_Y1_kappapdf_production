import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from settings import S


class GPR :
    

    def __init__ (self, data, reduction=np.mean, test_idx=None, subsample=None, kernel=None) :
        """ constructor
        data ... something that behaves like a Data instance (but can be something different, e.g. compressed)
                 Needs to implement get_cosmo and get_datavec methods
        reduction ... the function used to collapse over random seeds
        test_idx ... if given, remove this cosmology from the training set (for testing)
        subsample ... if given, use only <subsample> of the available augmented realizations
                      per cosmology (same at each cosmology)
        """

        # TODO I think there are cases where we don't do the normalization by yfid
        #      (because it's not possible or something)
        #      Could be for covariance matrix

        self.xfid = data.get_cosmo('fiducial')
        x = self._norm_x(data.get_cosmo('cosmo_varied'))

        self.yfid = reduction(data.get_datavec('fiducial'), axis=0)
        y = data.get_datavec('cosmo_varied')

        if subsample is not None :
            assert y.shape[1] == data.get_nseeds('cosmo_varied')
            # NOTE the implementation of this random sampling is duplicated in
            #      trial_chains.py, so be careful!
            select = np.random.default_rng(subsample)\
                        .choice(y.shape[1], size=subsample, replace=False)
            y = y[:, select, ...]

        y = reduction(y, axis=1) / self.yfid[None, ...] - 1

        if test_idx is not None :
            x = np.delete(x, test_idx, axis=0)
            y = np.delete(y, test_idx, axis=0)

        if kernel is None :
            length_scale = 1 if 'rbf_length_scale' not in S else S['rbf_length_scale']
            kernel = kernels.RBF(length_scale=length_scale, length_scale_bounds='fixed')

        # make sure we get reproducible results by setting the seed!
        self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=137).fit(x, y)


    def __call__ (self, x) :
        x = self._norm_x(x.reshape(-1, 2))
        y = self.gpr.predict(x).squeeze()
        if self.yfid is not None :
            y = (1 + y) * self.yfid
        return y


    def _norm_x (self, x) :
        """ center the inputs """
        return x - self.xfid[None, :]
