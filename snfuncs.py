import pickle
from os.path import isdir, isfile

from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

from utils import *

timeclipdict = {'SN2006aj': (2.5, np.inf)}  # from the original time count, that in the pycoco output
timeclip_sincemax = (-20, 60)  # time count here is time since max
LAMB = np.arange(4000, 8000, 20)  # AA

exclude_row_and_col = False  # only affects reading pickled file (True when e.g. features=distances)


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

        bounds, x0 = [(-1, 1), (0.9, 1.1)], [0, 1]

        self.minres = minimize(self.objective, bounds=bounds, x0=np.array(x0))
        self.result = self.minres.fun

        if np.isnan(self.result):
            raise Exception(sn1.name + ', ' + sn2.name)

    def transform(self, x):
        timeshift, fluxscale = x

        sn1_trans = SN()
        sn1_trans.name = self.sn1.name
        sn1_trans.time = self.sn2.time + timeshift
        sn1_trans.flux = self.sn1_flux_func(sn1_trans.time)
        sn1_trans.lamb = self.sn1.lamb

        sn2_trans = SN()
        sn2_trans.name = self.sn2.name
        sn2_trans.time = self.sn2.time
        sn2_trans.flux = fluxscale * self.sn2.flux
        sn2_trans.lamb = self.sn2.lamb

        return sn1_trans, sn2_trans

    def objective(self, x):
        sn1_trans, sn2_trans = self.transform(x)
        dif = sn1_trans.flux - sn2_trans.flux
        rms = np.sqrt(np.nanmean(dif ** 2))
        return rms

    def plot(self):
        sn1_trans, sn2_trans = self.transform(self.minres.x)
        title = 'RMS' + r'$\Delta f$ = ' + "{:.2E}".format(self.result)
        specalbum([sn1_trans, sn2_trans], labels=[sn1_trans.name, sn2_trans.name + ' (fitted)'], title=title)


def calcfeatures(snlist):
    """
    function which takes snlist and edits sn.features for all sn in snlist (only used when creating)
    """

    TIME = np.arange(*timeclip_sincemax, 1)  # days since max
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
