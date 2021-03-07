from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

"""train"""
snlist, X = sne_list(exc)  # load/create the list of SNe from the info_df
X_PC, m, scaler = empca_(X, n_components=10, niter=15)
dismat, build_dissimilarity_matrix = unsup_rf(X_PC)
# At this point training is completed and its output, dismat (along with the metadata in snlist), is used for analysis.

"""view results"""
# dismatplot(dismat, snlist)  # view dissimilarity matrix

# dismat_emb = tsne_(dismat, n_components=3)
# dismat_emb = empca_(dismat_emb, n_components=3)  # rotate embeddding so that dimensions are PCs
# myscatter(dismat_emb, snlist, sfsize=False, save_anim=False)  # 2D or 3D scatter plot
# cornerplot(dismat_emb, snlist, sfsize=False)  # corner plot

# mstgraph(dismat, snlist)  # MST of the resulting dissimilarity matrix


"""query test SNe"""
query_names = ['SN2005bf', 'SN2008D', 'SN2017hyh', 'SN2011bm', 'SN2007ru', 'SN2016bkv', 'SN2012cg', 'SN2012au',
               'SN2004gt', 'SN2013gh']

dissims_to_training, query_snlist = query(query_names, X_PC, m, scaler)
dismatplot_query(dissims_to_training, query_snlist, snlist, dismat)  # view dissimilarities with training set
