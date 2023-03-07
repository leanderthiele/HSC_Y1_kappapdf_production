import numpy as np

from data import DataPart
from gpr import GPR
from settings import S

# NOTE this class doesn't work for 'joint' MOPED yet!

class Cov :

    def __init__ (self, data, test_idx=None) :
        """ constructor
        data is something that behaves publicly like a Data instance
        test_idx ... in case some emulations are performed
        """

        self.test_idx = test_idx
        
        self.fid_cov = np.cov(data.get_datavec('fiducial'), rowvar=False)
        self.fid_covinv = np.linalg.inv(self.fid_cov)
        self.fid_sigma = np.sqrt(np.diagonal(self.fid_cov))

        self.used_stats = data.get_used_stats()

        self.all_fixed = all(S[stat]['cov_mode'] == 'fixed' for stat in self.used_stats)
        if self.all_fixed : # easy case, nothing to do
            return

        # can populate this with further information which we may later need
        self.cov_blocks = {
                           stat: {'mode': S[stat]['cov_mode'], 'data': DataPart(data, stat), }
                           for stat in self.used_stats
                          }
        for c in self.cov_blocks.values() :
            self._prepare_cov_block(c)

        # correlation matrices between different statistics
        self.corr_blocks = self._corr_blocks()


    def cov (self, theta) :
        if self.all_fixed :
            return self.fid_cov
        diag_cov_blocks = {stat: self._eval_cov_block(c, theta) for stat, c in self.cov_blocks.items()}
        cov_blocks = []
        for stat1, cov1 in diag_cov_blocks.items() :
            cov_blocks.append([])
            for stat2, cov2 in diag_cov_blocks.items() :
                if stat1 == stat2 :
                    cov_blocks[-1].append(cov1)
                else :
                    scale1 = np.sqrt(np.diagonal(cov1))
                    scale2 = np.sqrt(np.diagonal(cov2))
                    cov_blocks[-1].append(self.corr_blocks[f'{stat1}_{stat2}'] \
                                          * (scale1[:, None] * scale2[None, :]))
        return np.block(cov_blocks)


    def covinv (self, theta) :
        return np.linalg.inv(self.cov(theta))


    def _corr_blocks (self) :
        out = {}
        for stat1, v1 in self.cov_blocks.items() :
            for stat2, v2 in self.cov_blocks.items() :
                if stat1 == stat2 :
                    continue
                x1 = v1['data'].get_datavec('fiducial')
                x2 = v2['data'].get_datavec('fiducial')
                corr = np.corrcoef(x1, x2, rowvar=False)
                out[f'{stat1}_{stat2}'] = corr[:x1.shape[-1], :][:, x1.shape[-1]:]
        return out


    @staticmethod
    def _cinv_reduction (x, axis) :
        """ needs to conform to signature of np.mean """

        # if axis==0, this generates a unit-dim leading axis,
        # otherwise the axes before axis are flattened
        x1 = x.reshape(-1, *x.shape[axis:])
        cinv = np.array([np.linalg.inv(np.cov(_x1, rowvar=False)) for _x1 in x1])
        # now reshape this back into required form (flattening the matrix into final dimension)
        cinv = cinv.reshape(*x.shape[:axis], -1)
        # Hartlap factor is maybe important here
        n = x.shape[axis]
        p = x.shape[-1]
        return cinv * (n-p-2)/(n-1)
    

    def _prepare_cov_block (self, cov_block) :
        """ pass a dict to which the necessary information will be added """

        if (mode := cov_block['mode']) in ['fixed', 'scale', ] :
            cov_block['cov'] = np.cov(cov_block['data'].get_datavec('fiducial'), rowvar=False)
            if mode == 'scale' :
                cov_block['sigma_fid'] = np.sqrt(np.diagonal(cov_block['cov']))
                cov_block['sigma_gpr'] = GPR(cov_block['data'], reduction=np.std,
                                             test_idx=self.test_idx)
        elif mode.startswith('gpr') :
            cov_block['cinv_gpr'] = GPR(cov_block['data'], reduction=Cov._cinv_reduction,
                                        test_idx=self.test_idx)
            if mode == 'gpr_scale' :
                sigma_fid = np.std(cov_block['data'].get_datavec('fiducial'), axis=0)
                gpr_cinv_at_fid = cov_block['cinv_gpr'](cov_block['data'].get_cosmo('fiducial'))
                d = int(np.round(np.sqrt(len(gpr_cinv_at_fid))))
                gpr_cov_at_fid = np.linalg.inv(gpr_cinv_at_fid.reshape(d, d))
                sigma_gpr = np.sqrt(np.diagonal(0.5 * (gpr_cov_at_fid + gpr_cov_at_fid.T)))
                cov_block['scale'] = sigma_fid / sigma_gpr
        else :
            raise NotImplementedError(mode)

    
    def _eval_cov_block (self, cov_block, theta) :

        if (mode := cov_block['mode']) == 'fixed' :
            out = cov_block['cov']
        elif mode == 'scale' :
            # this is quite hacky, but it seems to be better to keep Omega_matter fixed in this evaluation
            sigma = cov_block['sigma_gpr'](np.array([theta[0], cov_block['data'].get_cosmo('fiducial')[1]]))
            scaling = sigma / cov_block['sigma_fid']
            out = cov_block['cov'] * (scaling[:, None] * scaling[None, :])
        elif mode.startswith('gpr') :
            cinv = cov_block['cinv_gpr'](theta)
            d = int(np.round(np.sqrt(len(cinv))))
            c = np.linalg.inv(cinv.reshape(d, d))
            out = 0.5 * (c + c.T) # make sure symmetrical, probably not necessary
            if mode == 'gpr_scale' :
                out = out * (cov_block['scale'][:, None] * cov_block['scale'][None, :])
        else :
            raise NotImplementedError(mode)
        return out


# TESTING
if __name__ == '__main__' :
    from compressed_data import CompressedData
    cd = CompressedData()
    cov = Cov(cd)
    theta = np.array([0.82 * np.sqrt(0.279 / 0.3), 0.279])
    # theta = np.array([0.7, 0.4])
    cinv = cov.covinv(theta)
    with np.printoptions(precision=2, suppress=True, threshold=10000000, linewidth=200) :
        print(cov.fid_covinv)
        print(cinv)
