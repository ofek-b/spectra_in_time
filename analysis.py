import networkx as nx
from empca import empca
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as skTSNE
from sklearn.preprocessing import StandardScaler

from snfuncs import *  # noqa: F401 unused import
from utils import *


def train(onlymeta=False):
    exc = ['SN2005bf', 'SN2008D', 'SN2017hyh', 'SN2011bm', 'SN2007ru', 'SN2016bkv', 'SN1987A', 'SN2010al', 'SN2009ip',
           'SN2012cg', 'SN2012au', 'SN2004gt', 'SN2011fe', 'SN2013gh']  # exclude SNe
    exc += info_df[(info_df['Type'] == 'type to exclude')].index.to_list()  # exclude entire types

    snlist, X = sne_list(exc)  # load/create the list of SNe from the info_df
    if onlymeta:
        return exc, snlist
    X_PC, m, scaler = empca_(X, n_components=10, niter=15)
    dismat, build_dissimilarity_matrix = unsup_rf(X_PC)
    # Training is now completed and its output, dismat (along with the metadata in snlist), is used for analysis.

    return exc, snlist, X, X_PC, m, scaler, dismat, build_dissimilarity_matrix


def empca_(X, n_components=3, niter=25, showeigenvecs=False):
    X = X.copy()
    X[np.isnan(X)] = 0
    weights = ~np.isnan(X) + 0

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    m = empca(X, weights, nvec=n_components, niter=niter)
    X_PC = m.coeff

    # loss = 1 - m.R2()

    if showeigenvecs:
        for i in range(n_components):
            f = np.reshape(m.eigvec[i], (len(TIME), len(LAMB)))
            plt.contourf(TIME, LAMB, f.T, 100)
            plt.title('PC%s' % (i + 1))
            plt.xlabel('MJD - max ' + r'$m_B$')
            plt.ylabel(r'wavelength [$\AA$]')
            plt.colorbar(label=r'$f_\lambda$, scaled')
            plt.show()

    # if query:
    #     return X_PC, m, scaler
    return X_PC, m, scaler


def unsup_rf(X, N_TRAIN=500):
    """
    source: https://github.com/dalya/WeirdestGalaxies
    """

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
    # N_TRAIN = 500  # number of trees in the forest
    rand_f = RandomForestClassifier(n_estimators=N_TRAIN)
    rand_f.fit(X_total, Y_total)

    def build_dissimilarity_matrix(X):
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

        dismat = 1 - sim_mat
        print()
        print('dismat shape:', dismat.shape)

        nandiag = dismat.copy()
        np.fill_diagonal(nandiag, np.nan)
        print('median abs of symmetric difference, no diag terms =', np.nanmedian(np.abs(nandiag - nandiag.T)))
        sym_dismat = (dismat + dismat.T) / 2

        print('dismat diag:', np.unique(np.diag(sym_dismat)))
        underdiag = [sym_dismat[i, j] for i in range(sym_dismat.shape[0]) for j in range(i)]
        print('# of cells under diag:', int((sym_dismat.shape[0] ** 2 - sym_dismat.shape[0]) / 2), ', # of unique:',
              len(np.unique(underdiag)))
        print('dismat follows triangle inequality:', istriang(sym_dismat))

        return sym_dismat

    return build_dissimilarity_matrix(X), build_dissimilarity_matrix


def tsne_(X, n_components, perplexity=10, learning_rate=10, early_exaggeration=12):
    X = (X + X.T) / 2
    np.fill_diagonal(X, 0)
    tsn = skTSNE(n_components=n_components, metric='precomputed', perplexity=perplexity, learning_rate=learning_rate,
                 method='exact', early_exaggeration=early_exaggeration)
    X_emb = tsn.fit_transform(X)
    print('tSNE K-L divergence:', tsn.kl_divergence_)
    return X_emb


def mstgraph(dismat, snlist, onlytypes=None):
    # if dismat is None:
    #     snlist, X = sne_list(exc)  # load/create the list of SNe from the info_df
    #
    #     # non-standard: train only on selected SNe:
    #     # if onlytypes is not None:
    #     #     idxs = [i for i, sn in enumerate(snlist) if sn.type in onlytypes]
    #     #     snlist = [snlist[i] for i in idxs]
    #     #     # calc_eqw(snlist, (6500,6660))
    #
    #     X_PC = empca_(X, n_components=10, niter=15)
    #     dismat, _ = unsup_rf(X_PC)
    # elif snlist is None:
    #     raise Exception('Either enter dismat and snlist or none of them.')

    # after training on all SNe (standard procedure), exclude some SNe merely from view
    if onlytypes is not None:
        idxs = [i for i, sn in enumerate(snlist) if sn.type in onlytypes]
        snlist = [snlist[i] for i in idxs]
        dismat = dismat[:, idxs][idxs, :]

    g = nx.Graph(dismat)
    mst = nx.minimum_spanning_tree(g)

    plot_mst(mst, snlist)


def query(query_names):
    exc, snlist, X, X_PC, m, scaler, dismat, build_dissimilarity_matrix = train()  # train using the template SNe

    dissims_to_training, query_snlist = [], []
    for nm in query_names:
        query_sn = SN(nm)
        query_sn.type = query_sn.name + ' (query)'
        query_snlist.append(query_sn)
        query_x = calcfeatures([query_sn])
        query_x[np.isnan(query_x)] = 0
        query_weights = ~np.isnan(query_x) + 0

        # apply trained rf on query SN:
        query_x = scaler.transform(query_x.reshape(1, -1))
        m.set_data(query_x, query_weights)
        query_x_PC = m.coeff
        dismat_q = build_dissimilarity_matrix(np.row_stack([X_PC, query_x_PC]))
        dissims_to_training.append(dismat_q[-1, :-1])

    return dissims_to_training, query_snlist, snlist, dismat