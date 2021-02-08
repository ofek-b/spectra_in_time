from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

exc += info_df[(info_df['Type'] == '')].index.to_list()  # if you want to further exclude more SNe, or entire types
snlist = sne_list(exc)  # load/create the list of SNe from the info_df

red = Empca(snlist, n_components=10, niter=15)

confm, build_similarity_matrix = unsup_rf(red.reduced_data)  # just writing the PCs into the class instances.
for i, sn in enumerate(snlist):
    sn.features = confm[i, :]

snconfmtx(snlist, (5, 50))  # view dissimilarity matrix

red2 = TSNE(snlist, n_components=3)
# red2.plotloss()
print(red2.loss)
red2.show(sfsize=False)  # 2D or 3D scatter plot
red2.showc(getpcs=False)  # corner plot
