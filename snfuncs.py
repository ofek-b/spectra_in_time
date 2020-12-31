import pickle
from os.path import isdir, isfile

from joblib import Parallel, delayed
from pyemd.emd import emd
from tqdm import tqdm

from utils import *

timeclipdict = {'SN2006aj': (2.5, np.inf)}  # from the original time count, that in the pycoco output
timeclip_sincemax = (-20, 60)  # time count here is time since max
LAMB = np.arange(4000, 8000, 20)  # AA

exclude_row_and_col = False  # only affects reading pickled file (True when e.g. features=distances)
exc = ['SN2005bf', 'SN2008D', 'SN2017hyh', 'SN2011bm', 'SN2007ru', 'SN2016bkv', 'SN1987A', 'SN2010al', 'SN2009ip',
       'SN2012cg', 'SN2012au']


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
        plt.xlabel('MJD - max ' + r'$m_B$')
        plt.ylabel(lambstr)
        plt.colorbar(label=r'$f_\lambda$, scaled')
        plt.show()


class Dissimilarity:
    def __init__(self, sn1, sn2):
        assert np.all(sn1.lamb == sn2.lamb)
        self.sn1, self.sn2 = sn1, sn2

        self.sn1_flux_func = interp1d(self.sn1.time, self.sn1.flux, axis=0, fill_value=np.nan, bounds_error=False)
        self.sn2_flux_func = interp1d(self.sn2.time, self.sn2.flux, axis=0, fill_value=np.nan, bounds_error=False)

        time = np.arange(*timeclip_sincemax, 2)

        self.sn1_trans = SN()
        self.sn1_trans.name = self.sn1.name
        self.sn1_trans.time = time
        self.sn1_trans.flux = self.sn1_flux_func(self.sn1_trans.time)
        self.sn1_trans.lamb = self.sn1.lamb

        self.sn2_trans = SN()
        self.sn2_trans.name = self.sn2.name
        self.sn2_trans.time = time
        self.sn2_trans.flux = self.sn2_flux_func(self.sn2_trans.time)
        self.sn2_trans.lamb = self.sn2.lamb

        # for i in range(sn1_trans.flux.shape[0]):
        #     for j in range(sn1_trans.flux.shape[1]):
        #         if np.isnan(sn1_trans.flux[i,j]) or np.isnan(sn2_trans.flux[i,j]):
        #             sn1_trans.flux[i, j] = 0
        #             sn2_trans.flux[i, j] = 0

        fl1, fl2 = self.sn1_trans.flux.flatten(), self.sn2_trans.flux.flatten()
        fl1 = np.array(fl1, dtype='float64')
        fl2 = np.array(fl2, dtype='float64')
        keep = ~np.isnan(fl1) * ~np.isnan(fl2)
        fl1, fl2 = fl1[keep], fl2[keep]
        fl1 /= np.sum(fl1)
        fl2 /= np.sum(fl2)
        #
        # print(fl1.shape,fl2.shape)

        T = np.column_stack([self.sn2_trans.time.T for _ in range(len(self.sn2_trans.lamb))]).flatten()
        L = np.row_stack([self.sn2_trans.lamb for _ in range(len(self.sn2_trans.time))]).flatten()
        t, l = T[keep], L[keep]
        tm1, tm2 = np.meshgrid(t, t)
        lm1, lm2 = np.meshgrid(l, l)
        self.grounddist = (np.abs(lm1 - lm2) + 100 * np.abs(tm1 - tm2)) / 1000
        self.grounddist = np.array(self.grounddist, dtype='float64')

        self.result = emd(fl1, fl2, distance_matrix=self.grounddist)

    def plot(self):
        title = 'RMS' + r'$\Delta f$ = ' + "{:.2E}".format(self.result)
        specalbum([self.sn1_trans, self.sn2_trans], labels=[self.sn1_trans.name, self.sn2_trans.name + ' (fitted)'],
                  title=title)


TIME = np.arange(*timeclip_sincemax, 1)  # days since max


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
