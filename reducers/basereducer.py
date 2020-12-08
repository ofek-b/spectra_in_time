from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotting
import numpy as np


class BaseReducer(ABC):
    def __init__(self, snlist, n_components):
        self.snlist = snlist
        self.n_components = n_components

        # self.data = np.row_stack([sn.features for sn in snlist])
        # self.types = [sn.type for sn in snlist]
        # self.colors = [typ2color[sn.type] for sn in snlist]
        # self.names = [sn.name for sn in snlist]

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

    def show(self,dims=None,hidetypes=[]):
        if dims is None and self.n_components not in [2, 3]:
            raise Exception('n_components must be 2 or 3 for show. Use showc instead. You can come back and specifiy dims, len(dims)=2 or 3, for show.')
        if dims is not None and len(dims) not in [2,3]:
            raise Exception('len(dims) must be 2 or 3.')
        pcs = self.get_pcs(self.n_components)
        plotting.myscatter(pcs, self.snlist,dims)

    def showc(self,dims=None):
        pcs = self.get_pcs(self.n_components)
        plotting.cornerplot(pcs,self.snlist,dims)


def metricloss(distmatrix, reduced_data):
    sum_d2 = 0
    for i in range(reduced_data.shape[0]):
        for j in range(i + 1, reduced_data.shape[0]):
            sum_d2 += abs(distmatrix[i, j] - np.sqrt(sum((reduced_data[i, :] - reduced_data[j, :]) ** 2))) / distmatrix[
                i, j]

    return sum_d2 / (reduced_data.shape[0] * (reduced_data.shape[0] - 1) / 2)
