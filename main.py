from analysis import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

"""view results"""
exc, snlist, X, X_PC, m, scaler, dismat, build_dissimilarity_matrix = train()

dismatplot(dismat, snlist)  # view dissimilarity matrix

dismat_emb = tsne_(dismat, n_components=3)
# dismat_emb = empca_(dismat_emb, n_components=3)  # rotate embeddding so that dimensions are PCs
myscatter(dismat_emb, snlist, sfsize=False, save_anim=False)  # 2D or 3D scatter plot
# cornerplot(dismat_emb, snlist, sfsize=False)  # corner plot

# mstgraph(dismat, snlist)  # MST of the resulting dissimilarity matrix

"""query test SNe"""
query_names = ['SN2005bf', 'SN2008D', 'SN2017hyh', 'SN2011bm', 'SN2007ru', 'SN2016bkv', 'SN2012cg', 'SN2012au',
               'SN2004gt', 'SN2013gh']

dissims_to_training, query_snlist ,snlist, dismat= query(query_names)
dismatplot_query(dissims_to_training, query_snlist, snlist, dismat)  # view dissimilarities with training set
