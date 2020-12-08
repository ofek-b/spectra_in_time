import numpy as np  # noqa
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm

from config import *
from dissimilarity.timeshift_scaleflux import Dissimilarity
from pycoco2SNobject import sne_list, SN  # noqa: F401 unused import
from reducers.empcaa import empcaa  # noqa: F401 unused import
from reducers.isomap import Isomap  # noqa: F401 unused import
from reducers.mds import MDS  # noqa: F401 unused import


def ftrs_dissim(snlist_):
    for sn in tqdm(snlist_):
        sn.features = Parallel(n_jobs=NUM_JOBS, verbose=0)(
            delayed(lambda sn2: Dissimilarity(sn, sn2).result)(sn2) for sn2 in snlist_)


def ftrs_longline(snlist_):
    TIME = np.arange(-20, 60, 1)  # days since max
    for sn in tqdm(snlist_):
        iflux = interp1d(sn.time, sn.flux, axis=0, bounds_error=False, fill_value=np.nan)
        iflux = iflux(TIME)
        sn.features = iflux.flatten()


# todo: timestrech, varflux
# todo: longline of derivatives (weighted mean by actual flux?)
# todo: only mangled spectra
if __name__ == '__main__':
    # print(info_df['Type'].value_counts())

    # tr, tp = SN('SN2013am'), SN('SN2013fs')
    # tr.specalbum()
    # tp.specalbum()
    # tr.specalbum()
    # Dissimilarity(tr,tp).plot()
    # for nm in ['SN2006aj','SN1998bw','SN2014L']:
    #     sn=SN(nm)
    #     sn.surface()

    snlist = sne_list(snlistname="67+9Ia+2Ic+1IcBL+4Ib+2IIb, rebinned20, dof - 20days on maxB+10pc on maxnrmlz, rms",
                      calcfeatures=ftrs_dissim,
                       sne_to_exclude=[

                                         'SN1987A','SN2005bf', 'SN2009ip','SN2006aj'
                                       'SN2011bm', 'SN2013am', 'SN2009jf', 'SN2008D'
                                       ]
                                     # +info_df[(info_df['Type']=='7')| (info_df['Type']=='II') |(info_df['Type']=='7')].index.to_list()
                      ,
                      exclude_row_and_col=False)

    red = empcaa(snlist, n_components=13, n_neighbors=5, niter=30)
    # red.plotloss()
    # red.show([8, 6, 10])
    red.showc()
