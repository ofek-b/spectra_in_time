from reduction import snconfmtx  # noqa: F401 unused import
from snfuncs import SN, Dissimilarity, sne_list  # noqa: F401 unused import


exc=['SN1987A', 'SN2005bf', 'SN2009ip', 'SN2011bm', 'SN2013am', 'SN2009jf', 'SN2008D']
snlist = sne_list(exc)
snconfmtx(snlist)


# tr, tp = SN('SN2007af'), SN('SN2012ht')
# tr.specalbum()
# tp.specalbum()
# Dissimilarity(tr, tp).plot()


# def ftrs_dissim(snlist_):
#     for sn in tqdm(snlist_):
#         sn.features = Parallel(n_jobs=NUM_JOBS, verbose=0)(
#             delayed(lambda sn2: Dissimilarity(sn, sn2).result)(sn2) for sn2 in snlist_)
#
#
# def ftrs_longline(snlist_):
#     TIME = np.arange(-20, 60, 1)  # days since max
#     for sn in tqdm(snlist_):
#         iflux = interp1d(sn.time, sn.flux, axis=0, bounds_error=False, fill_value=np.nan)
#         iflux = iflux(TIME)
#         sn.features = iflux.flatten()
