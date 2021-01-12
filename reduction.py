from abc import ABC, abstractmethod

from empca import empca
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import Isomap as skIsomap
from sklearn.manifold import MDS as skMDS
from sklearn.manifold import TSNE as skTSNE
from sklearn.preprocessing import StandardScaler

from snfuncs import TIME, LAMB
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

    def show(self, dims=None, hidetypes=[], getpcs=True):
        if dims is None and self.n_components not in [2, 3]:
            raise Exception(
                'n_components must be 2 or 3 for show. Use showc instead. You can come back and specifiy dims, len(dims)=2 or 3, for show.')
        if dims is not None and len(dims) not in [2, 3]:
            raise Exception('len(dims) must be 2 or 3.')
        pcs = self.get_pcs(self.n_components) if getpcs else self.reduced_data
        myscatter(pcs, self.snlist, dims)

    def showc(self, dims=None, getpcs=True):
        pcs = self.get_pcs(self.n_components) if getpcs else self.reduced_data
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
        self.m = empca(data, weights, nvec=self.n_components, niter=self.niter)
        reduced_data = self.m.coeff

        loss = 1 - self.m.R2()
        return reduced_data, loss

    def showeigenvecs(self):
        for i in range(self.n_components):
            f = np.reshape(self.m.eigvec[i], (len(TIME), len(LAMB)))
            plt.contourf(TIME, LAMB, f.T, 100)
            plt.title('PC%s' % (i + 1))
            plt.xlabel('MJD - max ' + r'$m_B$')
            plt.ylabel(lambstr)
            plt.colorbar(label=r'$f_\lambda$, scaled')
            plt.show()

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


class TSNE(BaseReducer):
    def __init__(self, snlist, n_components=3, perplexity=10, learning_rate=10, **kwargs):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        super().__init__(snlist, n_components=n_components)

    def reducedim(self):
        data = np.row_stack([sn.features for sn in self.snlist])
        data = (data + data.T) / 2
        np.fill_diagonal(data, 0)
        tsn = skTSNE(n_components=self.n_components, metric='precomputed', perplexity=self.perplexity,
                     learning_rate=self.learning_rate, method='exact')
        reduced_data = tsn.fit_transform(data)
        loss = tsn.kl_divergence_
        return reduced_data, loss

    def optimparam(self):
        return 'perplexity', range(1, 50)


def snconfmtx(snlist, cbar_prcntls=(5, 50)):
    fulltypes = [info_df['FullType'][sn.name] if not pd.isna(info_df['FullType'][sn.name]) else info_df['Type'][sn.name]
                 for sn in snlist]

    order = defaultdict(lambda: 0, {'Ia': 7, 'Ib': 4, 'Ibc': 4.5, 'Ic': 5, 'Ic-BL': 6, 'II': 2, 'IIn': 1, 'IIb': 3})
    snlist_ftyp_idxs = sorted([i for i, ftyp_ in enumerate(fulltypes)], key=lambda i: order[fulltypes[i]])

    snlist_ftyp = [snlist[i] for i in snlist_ftyp_idxs]

    mtx = np.row_stack([np.array(sn.features)[snlist_ftyp_idxs] for sn in snlist_ftyp])
    mtx = (mtx + mtx.T) / 2
    for i in range(mtx.shape[0]):
        for j in range(i, mtx.shape[1]):
            mtx[i, j] = np.nan
    norm = LogNorm(np.nanpercentile(mtx, cbar_prcntls[0]), np.nanpercentile(mtx, cbar_prcntls[1]))
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


def unsup_rf(X):
    def return_synthetic_data(X):
        """
        The function returns a matrix with the same dimensions as X but with synthetic data
        based on the marginal distributions of its featues
        """
        features = len(X[0])
        X_syn = np.zeros(X.shape)

        for i in range(features):
            obs_vec = X[:, i]
            syn_vec = np.random.choice(obs_vec, len(
                obs_vec))  # here we chose the synthetic data to match the marginal distribution of the real data
            X_syn[:, i] += syn_vec

        return X_syn

    X_syn = return_synthetic_data(X)

    # Now lets plot the *'real'* and *'synthetic'* data to examine the properties of the *'synthetic'* data:

    # In[52]:

    # Now lets plot the marginal distributions of the *'real'* and *'synthetic'* data and make sure that they match for a given feature:

    # In[53]:

    # **Step 2:** Once we have the *'real'* and *'synthetic'* data, we merge them into a single sample and assign classes to each of these. We will then train an RF to distinguish between the different classes.
    # This step essentially converts the problem from unsupervised to supervised, since we have labels for the data.
    #
    # We train the forest on the **entire** data, without dividing it to training, validation, and test sets as typically done in supervised learning. This is because we do not need to test the algorithms performance on new data, but we need it to learn as much as possible from the input (*'real'*) data in order to detect outliers.
    #
    # In this demo we **do not perform parallel training** since the sample is small. In case of parallel training one must:
    # * Select a random subset of objects from X.
    # * Select a random subset of features from X.
    # * Build *'synthetic'* data with the same dimensions as the subset of the *'real'* data.
    # * Train N decision trees for this *'real'* and *'synthetic'* data.
    # * Repeat the process M times, each time with a new subset of objects and features.
    # * Merge all the decision trees into a forest, the forest will contain NxM decision trees.

    # In[54]:

    def merge_work_and_synthetic_samples(X, X_syn):
        """
        The function merges the data into one sample, giving the label "1" to the real data and label "2" to the synthetic data
        """
        # build the labels vector
        Y = np.ones(len(X))
        Y_syn = np.ones(len(X_syn)) * 2

        Y_total = np.concatenate((Y, Y_syn))
        X_total = np.concatenate((X, X_syn))
        return X_total, Y_total

    X_total, Y_total = merge_work_and_synthetic_samples(X, X_syn)
    # declare an RF
    N_TRAIN = 500  # number of trees in the forest
    rand_f = RandomForestClassifier(n_estimators=N_TRAIN)
    rand_f.fit(X_total, Y_total)

    # Lets plot the probability of an object, which is described by the coordiantes (Feature 1, Feature 2), to be classified as *'real'* by the trained RF. This will give us a sense of the fitting that is done.

    # In[55]:

    # One can see that in the parameter range of Feature 1: 40-80, and Feature 2: 40-100, the classifier performs well and is able to describe the boundries of the *'real'* data well. This is not true outside this range, since we do not have *'real'* data there. However, this is not an issue since we wish to detect outliers where *'real'* actually exists.

    # **Step 3:** Having a trained RF, we now build the similarity matrix that describes the pair-wise similarity of all the *'real'* objects in our sample. We note that from this point, we do not need the *'synthetic'* data any more.
    #
    # The algorithm presented by Shi & Horvath (2006) propagates each pair of objects in the decision trees and counts how many times these objects ended up in the same terminal node (leaf). Since a leaf in the tree describes the same route inside the tree, objects that end up in the same leaf are described by the same model within the same tree and therefore are similar. The similarity measure can vary from 0 (objects never end up in the same leaf) to N_trees (objects ended up in the same leaf in all the decision trees).
    #
    # The next cell shows the schematic process of measuring the similarity measure:

    # In[19]:

    # In our case, we find that counting all the leafs, regardless of their prediction, is not optimal.
    # Instead, we propagate the objects through the decision trees and count how many times these objects ended up in the same leaf which **ALSO** predicts both of the objects to be real.
    # In our demo example this does not change the outliers that we get, but for outlier detection on galaxy spectra we find that galaxies with very low signal-to-noise ratio (essentially no signal detected), which are often predicted to be *'synthetic'* by the algorithm, add significant noise to the final distance measure.
    # In the next cells we compute the similarity matrix when taking into account only leaves that predict both of the objects to be *'real'*:

    # In[56]:

    def build_similarity_matrix(X):
        """
        The function builds the similarity matrix based on the feature matrix X for the results Y
        based on the random forest we've trained
        the matrix is normalised so that the biggest similarity is 1 and the lowest is 0

        This function counts only leaves in which the object is classified as a "real" object
        it is also implemented to optimize running time, asumming one has enough running memory
        """
        # apply to get the leaf indices
        apply_mat = rand_f.apply(X)
        # find the predictions of the sample
        is_good_matrix = np.zeros(apply_mat.shape)
        for i, est in enumerate(rand_f.estimators_):
            d = est.predict_proba(X)[:, 0] == 1
            is_good_matrix[:, i] = d
        # mark leaves that make the wrong prediction as -1, in order to remove them from the distance measurement
        apply_mat[is_good_matrix == False] = -1
        # now calculate the similarity matrix
        sim_mat = np.sum(
            (apply_mat[:, None] == apply_mat[None, :]) & (apply_mat[:, None] != -1) & (apply_mat[None, :] != -1),
            axis=2) / np.asfarray(np.sum([apply_mat != -1], axis=2), dtype='float')
        return sim_mat

    sim_mat = build_similarity_matrix(X)
    dis_mat = 1 - sim_mat

    return dis_mat, build_similarity_matrix
