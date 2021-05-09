from empca import empca
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as skTSNE
from sklearn.preprocessing import StandardScaler

from snfuncs import *


def train(onlymeta=False,additional_exc=[]):
    # exc = ['SN1987A', 'SN2009ip', 'SN2010al', 'SN2005bf', 'SN2013gh',
    #        'SN2011bm', 'SN2016bkv', 'SN2017hyh', 'SN2012cg', 'SN2012au', 'SN2004gt', 'SN2007ru',
    #        'SN2008D']  # old

    exc = []
    exc += additional_exc
    exc += ['slimSN2011bm', 'SN1991T', 'SN2011ke', 'SN2013fsV19']  # technical
    exc += [nm for nm in info_df.index.to_list() if '_' in nm]  # technical (artificially degraded)
    exc += ['SN1987A', 'SN2009ip', 'SN2005bf', 'SN2013gh', 'SN2011bm', 'SN2016bkv', 'SN2017hyh', 'SN2012cg',
            'SN2008D']  # avishay
    exc += ['SN2005cp', 'SN2008in', 'SN2009dd', 'SN2008aq', 'SN2011ht', 'PTF12bho', 'SN2018kjy',
            'SN2019pxu']  # data quality (DQI < 0.8)
    # exc += info_df[(info_df['FullType'] == 'SLSN-I')].index.to_list()

    # exc += ['SN2016hnk',
    # 'SN2018gwo',
    # 'SN2019bkc',
    # 'SN2019ccm',
    # 'SN2019gau',
    # 'SN2019hty',
    # 'SN2019mjo',
    # 'SN2019txl',
    # 'SN2019yz',
    # 'PTF09dav',
    # 'PTF11kmb',]


    # exc += info_df[(info_df['Type'] == 'type to exclude')].index.to_list()  # exclude entire types

    snlist, X = sne_list(exc)  # load/create the list of SNe from the info_df
    if onlymeta:
        return exc, snlist

    # show_missing(X, TIME, LAMB)

    _, m, scaler = empca_(X, n_components=50, niter=15)
    X_PC = scaler.inverse_transform(m.model)
    build_dissimilarity_matrix, rand_f = unsup_rf(X_PC, n_estimators=2000)  # 2000, max_features=2
    # Training is now complete and its output, dismat (along with the metadata in snlist), is used for analysis.

    return exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f


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


def unsup_rf(X, **scikit_kws):
    """
    based on: https://github.com/dalya/WeirdestGalaxies

    Args:
        X: training data, (n_samples, n_features)
        scikit_kws: keyword arguments to be passed to sklearn.ensemble.RandomForestClassifier
    Returns:
        build_dissimilarity_matrix: a function that takes a data set and returns its symmetricized dissimilarity matrix
        rand_f: scikit-learn RandomForestClassifier object
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
    rand_f = RandomForestClassifier(n_jobs=NUM_JOBS, **scikit_kws)
    rand_f.fit(X_total, Y_total)

    def build_dissimilarity_matrix(X, info=True):
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
        sim_mat = \
            np.sum((apply_mat[:, None] == apply_mat[None, :]) & (apply_mat[:, None] != -1) & (apply_mat[None, :] != -1),
                   axis=2) / np.asfarray(np.sum([apply_mat != -1], axis=2), dtype='float')

        dismat = 1 - sim_mat
        if info:
            print()
            print('dismat shape:', dismat.shape)

            nandiag = dismat.copy()
            np.fill_diagonal(nandiag, np.nan)
            print('median abs of symmetric difference, no diag terms =', np.nanmedian(np.abs(nandiag - nandiag.T)))
        sym_dismat = (dismat + dismat.T) / 2

        if info:
            print('dismat diag:', np.unique(np.diag(sym_dismat)))
            underdiag = [sym_dismat[i, j] for i in range(sym_dismat.shape[0]) for j in range(i)]
            print('# of cells under diag:', int((sym_dismat.shape[0] ** 2 - sym_dismat.shape[0]) / 2), ', # of unique:',
                  len(np.unique(underdiag)))
            # print('dismat follows triangle inequality:', istriang(sym_dismat))

        return sym_dismat

    return build_dissimilarity_matrix, rand_f


def tsne_(X, n_components, perplexity=10, learning_rate=10, early_exaggeration=12):
    X = (X + X.T) / 2
    np.fill_diagonal(X, 0)
    tsn = skTSNE(n_components=n_components, metric='precomputed', perplexity=perplexity, learning_rate=learning_rate,
                 method='exact', early_exaggeration=early_exaggeration)
    X_emb = tsn.fit_transform(X)
    print('tSNE K-L divergence:', tsn.kl_divergence_)
    return X_emb


def mstgraph(dismat, snlist, onlytypes=None):
    if onlytypes is not None:
        idxs = [i for i, sn in enumerate(snlist) if sn.type in onlytypes]
        snlist = [snlist[i] for i in idxs]
        dismat = dismat[:, idxs][idxs, :]

    g = networkx.Graph(dismat)
    mst = networkx.minimum_spanning_tree(g)

    plot_mst(mst, snlist)
