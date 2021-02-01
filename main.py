from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

exc += info_df[(info_df['Type'] == '') | (info_df['Type'] == '') | (info_df['Type'] == '')].index.to_list()

snlist = sne_list(exc)
red = Empca(snlist, n_components=10, niter=15)

confm, build_similarity_matrix = unsup_rf(red.reduced_data)
for i, sn in enumerate(snlist):
    sn.features = confm[i, :]

# snconfmtx(snlist, (5, 50))

red2 = TSNE(snlist, n_components=3)
# red2.plotloss()
print(red2.loss)
red2.show(getpcs=False, sfsize=True)

### query SN:

sn_q = SN('SN2012au')
calcfeatures(snlist + [sn_q])

red = Empca(snlist + [sn_q], n_components=10, niter=15)
confm, build_similarity_matrix = unsup_rf(red.reduced_data[:-1, :])
dists_q = 1 - build_similarity_matrix(red.reduced_data)

assert np.sum(np.abs(confm - dists_q[:-1, :-1])) == 0

ordered = sorted(zip(snlist, dists_q[-1, :]), key=lambda x: x[1])
for sn, d in ordered:
    print(sn.name, d)

for i, sn in enumerate(snlist + [sn_q]):
    sn.features = dists_q[i, :]

# snconfmtx(snlist, (10, 50))

red2 = TSNE(snlist + [sn_q], n_components=3)
# red2.plotloss()
print(red2.loss)
red2.show(getpcs=False)
