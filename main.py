from reduction import *  # noqa: F401 unused import
from snfuncs import *  # noqa: F401 unused import

exc += info_df[(info_df['Type'] == '7') | (info_df['Type'] == '7') | (info_df['Type'] == '7')].index.to_list()

snlist = sne_list(exc)
red = Empca(snlist, n_components=6, niter=15)
# red.plotloss()
# red.showeigenvecs()
red.showc(getpcs=False)
# red.show([1,6,7],getpcs=False)
