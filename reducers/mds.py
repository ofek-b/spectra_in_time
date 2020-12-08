import numpy as np
from sklearn.manifold import MDS as skMDS

from reducers.basereducer import BaseReducer, metricloss


class MDS(BaseReducer):
    def __init__(self, snlist, n_components=3,**kwargs):
        super().__init__(snlist, n_components=n_components)

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])
        data = (data + data.T) / 2
        np.fill_diagonal(data, 0)
        iso = skMDS(n_components=self.n_components, random_state=7, dissimilarity='precomputed', n_jobs=-1, n_init=100)
        reduced_data = iso.fit_transform(data)
        loss = metricloss(data, reduced_data)
        return reduced_data, loss

    def optimparam(self):
        return 'n_components', range(2, 26)
