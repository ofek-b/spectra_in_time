import numpy as np
from sklearn.preprocessing import StandardScaler

from empca import empca
from reducers.basereducer import BaseReducer


class empcaa(BaseReducer):
    def __init__(self, snlist, n_components=3, **kwargs):
        if 'niter' in kwargs:
            self.niter=kwargs['niter']
        else:
            self.niter=25
        super().__init__(snlist, n_components=n_components)

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])
        data[np.isnan(data)] = 0
        weights = ~np.isnan(data) + 0

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        m = empca(data, weights, nvec=self.n_components,niter=self.niter)
        reduced_data = m.coeff

        loss = 1 - m.R2()
        return reduced_data, loss

    def optimparam(self):
        return 'n_components', range(2, 9)
