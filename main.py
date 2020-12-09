from reduction import Isomap, MDS, Empca  # noqa: F401 unused import
from snfuncs import sne_list, SN  # noqa: F401 unused import


# exc=[]
exc = ['SN1987A', 'SN2005bf', 'SN2009ip',  'SN2011bm', 'SN2013am', 'SN2009jf', 'SN2008D'    ]


# +info_df[(info_df['Type']=='7')| (info_df['Type']=='II') |(info_df['Type']=='7')].index.to_list()

snlist = sne_list(exc)
red = MDS(snlist, n_components=13, n_neighbors=5, niter=30)
# red.plotloss()
red.show([1, 2, 4])
# red.showc()
