from empca import empca
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as skTSNE
from sklearn.preprocessing import StandardScaler

from snfuncs import TIME, LAMB
from utils import *


def empca_(X, n_components=3, niter=25, showeigenvecs=False):
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
            plt.ylabel(lambstr)
            plt.colorbar(label=r'$f_\lambda$, scaled')
            plt.show()

    return X_PC


def tsne_(X, n_components, perplexity=10, learning_rate=10, early_exaggeration=12):
    X = (X + X.T) / 2
    np.fill_diagonal(X, 0)
    tsn = skTSNE(n_components=n_components, metric='precomputed', perplexity=perplexity, learning_rate=learning_rate,
                 method='exact', early_exaggeration=early_exaggeration)
    X_emb = tsn.fit_transform(X)
    print('TSNE K-L divergence:', tsn.kl_divergence_)
    return X_emb


def unsup_rf(X):
    def return_synthetic_data(X):
        """
        source: https://github.com/dalya/WeirdestGalaxies
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
    N_TRAIN = 500  # number of trees in the forest
    rand_f = RandomForestClassifier(n_estimators=N_TRAIN)
    rand_f.fit(X_total, Y_total)

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
