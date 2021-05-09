from analysis import *

"""view results"""
exc, snlist, X, X_PC, m, scaler, build_dissimilarity_matrix, rand_f = train()
dismat = build_dissimilarity_matrix(X_PC)

sfpctgs = 1  # for showing superfit sizes in the graph set to not None
if sfpctgs is not None:
    sfpctgs, newnames = sfagreement(snlist)
    newidxs = [i for i, nm in enumerate([sn.name for sn in snlist]) if nm in newnames]
    snlist = [snlist[i] for i in newidxs]
    dismat = dismat[:, newidxs][newidxs, :]
    for sn in snlist:
        print(sn.name)

# dismatplot(dismat, snlist)  # view dissimilarity matrix


dismat_emb = tsne_(dismat, n_components=3, perplexity=10, learning_rate=10)
# dismat_emb = empca_(dismat_emb, n_components=3)  # rotate embeddding so that dimensions are PCs
# myscatter(dismat_emb, snlist, save_anim=False, legend=True, sfpctgs=sfpctgs)  # 2D or 3D scatter plot
cornerplot(dismat_emb, snlist, sfpctgs=sfpctgs)  # corner plot

# mstgraph(dismat, snlist)  # MST of the resulting dissimilarity matrix
