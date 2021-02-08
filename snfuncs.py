import pickle
from os.path import isdir, isfile

from joblib import Parallel, delayed
from tqdm import tqdm

from utils import *

"""input:"""

timeclipdict = {'SN2006aj': (2.5, np.inf)}  # from the original time count, that in the pycoco output
timeclip_sincemax = (-20, 60)  # time count here is time since max
LAMB = np.arange(4000, 8000, 20)  # AA

TIME = np.arange(*timeclip_sincemax, 1)  # days since max

exclude_row_and_col = False  # only affects reading pickled file (True when e.g. features=distances)
exc = ['SN2005bf', 'SN2008D', 'SN2017hyh', 'SN2011bm', 'SN2007ru', 'SN2016bkv', 'SN1987A', 'SN2010al', 'SN2009ip',
       'SN2012cg', 'SN2012au', 'SN2004gt', 'SN2011fe', 'SN2013gh']


class SN:
    def __init__(self, name=None):
        if name is None:
            return

        self.name = name
        self.features = None

        sedpath = PYCOCO_SED_PATH % self.name
        if isfile(sedpath):
            self.time, self.lamb, self.flux, self.fluxerr = read_pycoco_template_sed(sedpath, LAMB)
        else:
            pycoco_out_dir = PYCOCO_FINAL_DIR % self.name
            if isdir(pycoco_out_dir):
                self.time, self.lamb, self.flux, self.fluxerr = read_pycoco_template_outdir(pycoco_out_dir, LAMB)
            else:
                raise Exception('no template found for %s' % self.name)

        if self.name in timeclipdict:
            timeclip = timeclipdict[self.name]
            keep = (self.time >= timeclip[0]) * (self.time <= timeclip[1])
            self.time, self.flux, self.fluxerr = self.time[keep], self.flux[keep, :], self.fluxerr[keep, :]

        self.maxtime = timesincemax(self.time, self.lamb, self.flux, self.fluxerr)

        self.time = self.time - self.maxtime
        self.flux /= np.nanmax(self.flux)

        if timeclip_sincemax is not None:
            keep = (self.time >= timeclip_sincemax[0]) * (self.time <= timeclip_sincemax[1])
            self.time, self.flux, self.fluxerr = self.time[keep], self.flux[keep, :], self.fluxerr[keep, :]

        self.flux = np.gradient(np.log10(self.flux), axis=1)

    def specalbum(self):
        specalbum(self, labels=[self.name], title='0 d = ' + str(self.maxtime) + ' MJD')

    def surface(self):
        plt.contourf(self.time, self.lamb, self.flux.T, 100)
        plt.title(self.name + ', At Rest Frame After Host Corrections')
        plt.xlabel('days since B max')
        plt.ylabel(lambstr)
        plt.colorbar(label='Flux rescaled')
        plt.show()


def calcfeatures(snlist):
    """
    function which takes snlist and edits sn.features for all sn in snlist (only used when creating)
    """

    for sn in tqdm(snlist):
        iflux = interp1d(sn.time, sn.flux, axis=0, bounds_error=False, fill_value=np.nan)
        iflux = iflux(TIME)
        sn.features = iflux.flatten()


def sne_list(sne_to_exclude=None):
    """
    Parameters only take effect when reading the pickled snlist. Writing the file always takes all of the SNe.
    The pickled_snlist is read if it exists. Otherwise, it is created. The list of SNe is returned anyway.

    Args:
        sne_to_exclude: name list
    """

    if isfile(SNLIST_PATH):
        if sne_to_exclude is None or exclude_row_and_col is None:
            raise Exception('must enter sne_to_exclude (name list) and exclude_row_and_col (bool)')
        print('Reading pickled snlist')
        with open(SNLIST_PATH, 'rb') as f:
            snlist_ = pickle.load(f)

    else:
        snlist_ = Parallel(n_jobs=NUM_JOBS, verbose=0)(delayed(SN)(nm) for nm in info_df.index.to_list())
        # snlist_ = [SN(nm) for nm in tqdm(info_df.index.to_list())]
        calcfeatures(snlist_)
        with open(SNLIST_PATH, 'wb') as f:
            pickle.dump(snlist_, f)

    exclude_idxs = [i for i, sn in enumerate(snlist_) if sn.name in sne_to_exclude]
    snlist_ = [sn for i, sn in enumerate(snlist_) if i not in exclude_idxs]
    if exclude_row_and_col:
        for sn in snlist_:
            sn.features = [x for i, x in enumerate(sn.features) if i not in exclude_idxs]

    return snlist_
