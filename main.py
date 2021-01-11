from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

exc += info_df[(info_df['Type'] == '7') | (info_df['Type'] == '7') | (info_df['Type'] == '7')].index.to_list()

snlist = sne_list(exc)
red = Empca(snlist, n_components=8, niter=15)

confm = unsup_rf(red.reduced_data)
for i, sn in enumerate(snlist):
    sn.features = confm[i, :]

# snconfmtx(snlist)

red2 = MDS(snlist, n_components=3)
print(red2.loss)
red2.show(getpcs=False)
