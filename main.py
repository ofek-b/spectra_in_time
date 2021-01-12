from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

exc += info_df[(info_df['Type'] == '') | (info_df['Type'] == '') | (info_df['Type'] == '')].index.to_list()

snlist = sne_list(exc)
red = Empca(snlist, n_components=10, niter=15)

confm = unsup_rf(red.reduced_data)
for i, sn in enumerate(snlist):
    sn.features = confm[i, :]

# snconfmtx(snlist, (10, 50))

red2 = TSNE(snlist, n_components=3)
# red2.plotloss()
print(red2.loss)
red2.show(getpcs=False)
