from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

snlist, X = sne_list(exc)  # load/create the list of SNe from the info_df

X_PC = empca_(X, n_components=10, niter=15)
dismat, build_similarity_matrix = unsup_rf(X_PC)

# dismatplot(dismat, snlist)  # view dissimilarity matrix
dismat_emb = tsne_(dismat, n_components=3)
# dismat_emb = empca_(dismat_emb, n_components=3)  # rotate embeddding so that dimensions are PCs
myscatter(dismat_emb, snlist, sfsize=False, save_anim=True)  # 2D or 3D scatter plot
# cornerplot(dismat_emb, snlist, sfsize=False)  # corner plot
