import numpy as np

from data import Data, DataWrapper
from gpr import GPR
from settings import S


class DataPart (DataWrapper) :
    """ small helper that only picks out part of the data vector """

    def __init__ (self, data, stat) :
        super().__init__(data)
        assert stat in data.get_used_stats()
        self.stat = stat

    def get_datavec (self, case) :
        return self.data._get_data_array(self.stat, case)


class CompressedData (DataWrapper) :
    """ implement same public interface as Data """


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


    def get_datavec (self, case) :
        return np.concatenate([
                               np.einsum('ab, ...b->...a', p[1], p[0].get_datavec(case)) if p[1] is not None \
                               else p[0].get_datavec(case) \
                               for p in self.parts
                              ],
                              axis=-1)


    def _compression_weights (self, data) :
        """ the main functionality
        data should expose the usual interface
        Returns an array [Ncompressed, Nuncompressed]
        """

        Cinv = np.linalg.inv(np.cov(data.get_datavec('fiducial'), rowvar=False))
        dmdt = self._derivatives(data)
        b1 = np.einsum('ab,b->a', Cinv, dmdt[0]) \
             / np.sqrt(np.einsum('a,ab,b->', dmdt[0], Cinv, dmdt[0]))
        b2 = ( np.einsum('ab,b->a', Cinv, dmdt[1]) - np.einsum('a,a->', dmdt[1], b1) * b1 ) \
             / np.sqrt( np.einsum('a,ab,b->', dmdt[1], Cinv, dmdt[1]) - np.einsum('a,a->', dmdt[1], b1)**2 )
        return np.stack([b1, b2], axis=0)


    def _derivatives (self, data) :
        """ returns dmudtheta as a list """
        
        if (m := S['moped']['deriv_mode']) == 'gpr' :
            return self._derivatives_gpr(data)
        else :
            return self._derivatives_lstsq(data, int(m.split('_')[1]))

    
    def _derivatives_lstsq (self, data, N) :
        """ estimate using linear regression, N is the number of points to use """
        # TODO
        raise NotImplementedError()


    def _derivatives_gpr (self, data) :
        """ estimate from Gaussian process emulator """

        delta_theta = [0.05, 0.05] # TODO can play with this
        gpr = GPR(data)
        out = []
        for ii, delta in enumerate(delta_theta) :
            t = data.get_cosmo('fiducial')
            t[ii] += delta
            mu_hi = gpr(t)
            t[ii] -= 2*delta
            mu_lo = gpr(t)
            dmdt = (mu_hi-mu_lo)/(2*delta)
            out.append(dmdt)
        return out


# TESTING
if __name__ == '__main__' :
    
    cd = CompressedData()

    a = cd.get_datavec('fiducial')
    print(a.shape)
    b = cd.get_datavec('cosmo_varied')
    print(b.shape)

    print(np.cov(a, rowvar=False))
