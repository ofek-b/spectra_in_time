from analysis import *

"""view results"""
exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()
dismat = build_dissimilarity_matrix(X_PC)

# dismatplot(dismat, snlist)  # view dissimilarity matrix

dismat_emb = tsne_(dismat, n_components=3, perplexity=10, learning_rate=15)
# dismat_emb = empca_(dismat_emb, n_components=3)  # rotate embeddding so that dimensions are PCs
myscatter(dismat_emb, snlist, save_anim=False)  # 2D or 3D scatter plot
cornerplot(dismat_emb, snlist)  # corner plot

mstgraph(dismat, snlist)  # MST of the resulting dissimilarity matrix
