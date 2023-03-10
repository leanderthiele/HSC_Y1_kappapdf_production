import numpy as np

from sklearn.linear_model import LinearRegression

from data import Data, DataWrapper, DataPart
from gpr import GPR
from settings import S


class CompressedData (DataWrapper) :
    """ implement same public interface as Data """

    # use this covariance matrix to define a "close" region for the lstsq derivatives
    # this is not very rigorous but by eye it looks ok
    COV_LSTSQ = [[0.05**2,       0, ],
                 [0      , 0.10**2, ], ]


    def __init__ (self) :
        super().__init__(Data())
        if (a := S['moped']['apply_to']) == 'joint' :
            self.parts = [(self.data, self._compression_weights(self.data)), ]
        else :
            self.parts = [
                          (
                           d := DataPart(self.data, stat),
                           self._compression_weights(d) if stat in a else None
                          )
                          for stat in self.data.get_used_stats()
                         ]


    def get_datavec (self, case, stat=None) :
        if stat is not None :
            if S['moped']['apply_to'] == 'joint' :
                assert stat == 'joint'
                # and then fall though, since it'll work
            else :
                for p in self.parts :
                    if p[0].stat == stat :
                        return self._eval_compression(p, case)
                raise RuntimeError(stat)
        return np.concatenate([self._eval_compression(p, case) for p in self.parts ], axis=-1)


    def get_stat_mask (self, stat) :
        if S['moped']['apply_to'] == 'joint' :
            return np.full(self.parts[0][1].shape[0], stat=='joint', dtype=bool)
        out = []
        for p in self.parts :
            m = p[0].get_stat_mask(stat)
            if p[1] is None :
                out.append(m)
            else :
                assert np.all(m[0] == m)
                out.append(np.full(p[1].shape[0], m[0], dtype=bool))
        return np.concatenate(out)


    def get_used_stats (self) :
        if S['moped']['apply_to'] == 'joint' :
            return ['joint', ]
        return self.data.get_used_stats()


    def _compression_weights (self, data) :
        """ the main functionality
        data should expose the usual interface
        Returns an array [Ncompressed, Nuncompressed]
        """

        Cinv = np.linalg.inv(np.cov(data.get_datavec('fiducial'), rowvar=False))
        dmdt = self._derivatives(data)
        # with np.printoptions(precision=2, suppress=True, threshold=10000000, linewidth=200) :
        #     print(dmdt)
        b1 = np.einsum('ab,b->a', Cinv, dmdt[0]) \
             / np.sqrt(np.einsum('a,ab,b->', dmdt[0], Cinv, dmdt[0]))
        b2 = ( np.einsum('ab,b->a', Cinv, dmdt[1]) - np.einsum('a,a->', dmdt[1], b1) * b1 ) \
             / np.sqrt( np.einsum('a,ab,b->', dmdt[1], Cinv, dmdt[1]) - np.einsum('a,a->', dmdt[1], b1)**2 )
        return np.stack([b1, b2], axis=0)


    def _eval_compression (self, part, case) :
        return np.einsum('ab, ...b->...a', part[1], part[0].get_datavec(case)) if part[1] is not None \
               else part[0].get_datavec(case)


    def _derivatives (self, data) :
        """ returns dmudtheta as a list """
        
        if (m := S['moped']['deriv_mode']) == 'gpr' :
            return self._derivatives_gpr(data)
        else :
            return self._derivatives_lstsq(data, int(m.split('_')[1]))

    
    def _derivatives_lstsq (self, data, N) :
        """ estimate using linear regression, N is the number of points to use """

        delta_theta = data.get_cosmo('cosmo_varied') - data.get_cosmo('fiducial')[None, :]
        if 'compression_fid_shift' in S :
            delta_theta -= np.array(list(S['compression_fid_shift'].values()))
        Cinv_lstsq = np.linalg.inv(np.array(CompressedData.COV_LSTSQ))
        ds = np.einsum('ia,ab,ib->i', delta_theta, Cinv_lstsq, delta_theta)
        select = np.argsort(ds)[:N]
        delta_theta = delta_theta[select]
        y = np.mean(data.get_datavec('cosmo_varied')[select], axis=1)
        linear_model = LinearRegression(fit_intercept=True).fit(delta_theta, y)
        return linear_model.coef_.T # shape [params, data]


    def _derivatives_gpr (self, data) :
        """ estimate from Gaussian process emulator """

        # derivatives are remarkably stable under changes to this step size
        # (varations ~10 % when scanning step size between 0.01 and 0.1)
        delta_theta = [0.05, 0.05]
        gpr = GPR(data)
        out = []
        for ii, delta in enumerate(delta_theta) :
            t = data.get_cosmo('fiducial')
            if 'compression_fid_shift' in S :
                t += np.array(list(S['compression_fid_shift'].values()))
            t[ii] += delta
            mu_hi = gpr(t)
            t[ii] -= 2*delta
            mu_lo = gpr(t)
            dmdt = (mu_hi-mu_lo)/(2*delta)
            out.append(dmdt)
        return np.array(out)


# TESTING
if __name__ == '__main__' :
    
    cd = CompressedData()

#    a = cd.get_datavec('fiducial')
#    print(a.shape)
#    b = cd.get_datavec('cosmo_varied')
#    print(b.shape)

#    print(np.cov(a, rowvar=False))
