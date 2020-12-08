import numpy as np
from sklearn.manifold import Isomap as skIsomap

from reducers.basereducer import BaseReducer, metricloss


class Isomap(BaseReducer):
    def __init__(self, snlist, n_components=3, n_neighbors=10,metric='precomputed',**kwargs):
        self.n_neighbors = n_neighbors
        self.metric=metric
        super().__init__(snlist, n_components=n_components)

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])
        if self.metric=='precomputed':
            data = (data + data.T) / 2
            np.fill_diagonal(data, 0)
        iso = skIsomap(n_components=self.n_components, n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1)
        reduced_data = iso.fit_transform(data)
        loss = metricloss(data, reduced_data)
        return reduced_data, loss

    def optimparam(self):
        return 'n_neighbors', range(2, 50)
