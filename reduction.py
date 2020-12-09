from abc import ABC, abstractmethod

from empca import empca
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap as skIsomap
from sklearn.manifold import MDS as skMDS
from sklearn.preprocessing import StandardScaler



from utils import *


class BaseReducer(ABC):
    def __init__(self, snlist, n_components):
        self.snlist = snlist
        self.n_components = n_components
        self.reduced_data, self.loss = self.reducedim()

    @abstractmethod
    def reducedim(self):
        pass

    @abstractmethod
    def optimparam(self):
        pass

    def plotloss(self):
        param, rng = self.optimparam()
        losses = []
        for val in rng:
            d = {'n_components': self.n_components}
            d.update({param: val})
            loss = type(self)(self.snlist, **d).loss
            losses.append(loss)

        plt.plot(rng, losses, 's-')
        plt.xlabel(param)
        plt.ylabel('loss')
        plt.grid()
        plt.show()

    def get_pcs(self, view_dim):  # of reduced data, just rotation, no info loss
        scaler = StandardScaler()
        pca = PCA(n_components=self.reduced_data.shape[1])
        pcs = scaler.fit_transform(self.reduced_data)
        pcs = pca.fit_transform(pcs)
        pcs = scaler.inverse_transform(pcs)
        return pcs

    def show(self, dims=None, hidetypes=[]):
        if dims is None and self.n_components not in [2, 3]:
            raise Exception(
                'n_components must be 2 or 3 for show. Use showc instead. You can come back and specifiy dims, len(dims)=2 or 3, for show.')
        if dims is not None and len(dims) not in [2, 3]:
            raise Exception('len(dims) must be 2 or 3.')
        pcs = self.get_pcs(self.n_components)
        myscatter(pcs, self.snlist, dims)

    def showc(self, dims=None):
        pcs = self.get_pcs(self.n_components)
        cornerplot(pcs, self.snlist, dims)


def metricloss(distmatrix, reduced_data):
    sum_d2 = 0
    for i in range(reduced_data.shape[0]):
        for j in range(i + 1, reduced_data.shape[0]):
            sum_d2 += abs(distmatrix[i, j] - np.sqrt(sum((reduced_data[i, :] - reduced_data[j, :]) ** 2))) / distmatrix[
                i, j]

    return sum_d2 / (reduced_data.shape[0] * (reduced_data.shape[0] - 1) / 2)


class Empca(BaseReducer):
    def __init__(self, snlist, n_components=3, **kwargs):
        if 'niter' in kwargs:
            self.niter = kwargs['niter']
        else:
            self.niter = 25
        super().__init__(snlist, n_components=n_components)

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])
        data[np.isnan(data)] = 0
        weights = ~np.isnan(data) + 0

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        m = empca(data, weights, nvec=self.n_components, niter=self.niter)
        reduced_data = m.coeff

        loss = 1 - m.R2()
        return reduced_data, loss

    def optimparam(self):
        return 'n_components', range(2, 9)


class Isomap(BaseReducer):
    def __init__(self, snlist, n_components=3, n_neighbors=10, metric='precomputed', **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        super().__init__(snlist, n_components=n_components)

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])
        if self.metric == 'precomputed':
            data = (data + data.T) / 2
            np.fill_diagonal(data, 0)
        iso = skIsomap(n_components=self.n_components, n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1)
        reduced_data = iso.fit_transform(data)
        loss = metricloss(data, reduced_data)
        return reduced_data, loss

    def optimparam(self):
        return 'n_neighbors', range(2, 50)


class MDS(BaseReducer):
    def __init__(self, snlist, n_components=3, **kwargs):
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
