import random
from abc import ABC, abstractmethod
from random import shuffle

from empca import empca
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap as skIsomap
from sklearn.manifold import MDS as skMDS
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
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

    def show(self, dims=None, hidetypes=[],getpcs=True):
        if dims is None and self.n_components not in [2, 3]:
            raise Exception(
                'n_components must be 2 or 3 for show. Use showc instead. You can come back and specifiy dims, len(dims)=2 or 3, for show.')
        if dims is not None and len(dims) not in [2, 3]:
            raise Exception('len(dims) must be 2 or 3.')
        pcs = self.get_pcs(self.n_components) if getpcs else self.reduced_data
        myscatter(pcs, self.snlist, dims)

    def showc(self, dims=None,getpcs=True):
        pcs = self.get_pcs(self.n_components)  if getpcs else self.reduced_data
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


class SVM(BaseReducer):
    def __init__(self, snlist,logC=0, **kwargs):
        self.logC = logC
        self.fulltypes = [info_df['FullType'][sn.name] if not pd.isna(info_df['FullType'][sn.name]) else info_df['Type'][sn.name] for sn in snlist]
        super().__init__(snlist, n_components=len(set(self.fulltypes)))

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        svc=LinearSVC(C=np.exp(self.logC),max_iter=5000)
        svc.fit(data,self.fulltypes)  # sorted(self.fulltypes, key=lambda k: random.random())
        reduced_data = svc.decision_function(data)

        dct = dict(zip(svc.classes_, range(len(svc.classes_))))
        self.fulltypes_idx=[dct[typ] for typ in self.fulltypes]

        loss = -2*sum([reduced_data[i,typ] for i,typ in enumerate(self.fulltypes_idx) ]) + np.sum(reduced_data)

        return reduced_data,loss

    def optimparam(self):
        return 'logC', np.arange(-10,0,0.2)



def snconfmtx(snlist,bottomlimit=0.05,types=None):
    fulltypes = [info_df['FullType'][sn.name] if not pd.isna(info_df['FullType'][sn.name]) else info_df['Type'][sn.name]
                 for sn in snlist]
    snlist = [sn for ftyp,sn in zip(fulltypes,snlist) if types is None or ftyp in types]
    fulltypes = [info_df['FullType'][sn.name] if not pd.isna(info_df['FullType'][sn.name]) else info_df['Type'][sn.name]
                 for sn in snlist]

    order = {'Ia': 7, 'Ib': 4, 'Ic': 5, 'Ic-BL': 6, 'II': 2, 'IIn': 1, 'IIb': 3}
    snlist_ftyp_idxs = sorted([i for i, ftyp_ in enumerate(fulltypes)], key=lambda i: order[fulltypes[i]])

    snlist_ftyp = [snlist[i] for i in snlist_ftyp_idxs]

    mtx = np.row_stack([np.array(sn.features)[snlist_ftyp_idxs] for sn in snlist_ftyp])
    mtx = (mtx + mtx.T) / 2
    for i in range(mtx.shape[0]):
        for j in range(i, mtx.shape[1]):
            mtx[i, j] = np.nan
    norm = LogNorm(bottomlimit, np.nanmax(mtx))
    # norm = None
    plt.matshow(mtx, cmap='viridis', norm=norm)
    ax = plt.gca()

    typechange = [(ii + (ii - 1)) / 2 for ii in range(len(snlist_ftyp_idxs)) if
                  ii == 0 or fulltypes[snlist_ftyp_idxs[ii]] != fulltypes[snlist_ftyp_idxs[ii - 1]]]
    plt.vlines(typechange, -0.5, len(snlist_ftyp_idxs) - 0.5, colors='k', linestyles='dashed')
    plt.hlines(typechange, -0.5, len(snlist_ftyp_idxs) - 0.5, colors='k', linestyles='dashed')

    typechange = np.array(typechange + [len(snlist_ftyp_idxs) - 0.5])
    for p in (typechange[1:] + typechange[:-1]) / 2:
        typ = fulltypes[snlist_ftyp_idxs[int(p)]]
        plt.text(p, -2, typ, fontsize=11, color=typ2color[typ])
        plt.text(-5, p, typ, fontsize=11, color=typ2color[typ])

    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 50))
    ax.spines['top'].set_position(('outward', 50))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')

    ftypnames = [sn.name for sn in snlist_ftyp]
    plt.xticks(ticks=range(len(ftypnames)), labels=ftypnames, size=7, rotation=90)
    plt.yticks(ticks=range(len(ftypnames)), labels=ftypnames, size=7)
    plt.colorbar()
    # plt.tight_layout()
    plt.show()
