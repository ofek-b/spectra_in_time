import pickle
from glob import glob
from os.path import isfile, isdir, basename

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import simps
from scipy.interpolate import interp1d
from spectres import spectres

import plotting
from config import *


def timesincemax(time, lamb, flux, fluxerr):
    b = interp1d(band_wvl, band_throughput, fill_value=0, bounds_error=False)

    isnan = np.isnan(flux)
    flux_ = flux.copy()
    flux_[isnan] = 0
    synphots = np.average(flux_, axis=1, weights=b(lamb) * ~isnan)

    # synphots = simps(b(lamb) * flux, lamb)
    maxtime = time[np.nanargmax(synphots)]
    return maxtime


def read_pycoco_template_sed(sedpath):
    times, lambs, fluxes = zip(np.loadtxt(sedpath, comments='#', delimiter=' ', unpack=True))

    if isinstance(times, tuple):
        times = times[0]
    if isinstance(lambs, tuple):
        lambs = lambs[0]
    if isinstance(fluxes, tuple):
        fluxes = fluxes[0]

    dct = {}
    for t, lm, fl in zip(times, lambs, fluxes):
        if t in dct:
            dct[t].append((lm, fl))
        else:
            dct[t] = [(lm, fl)]

    fluxlst = []
    time = np.array(sorted(dct.keys()))
    for t in time:
        lsttup = sorted(dct[t], key=lambda tup: tup[0])
        lamb, flux = zip(*lsttup)

        flux = spectres(LAMB, np.array(lamb), np.array(flux), None, np.nan, False)

        fluxlst.append(flux)

    flux_matrix = np.row_stack(fluxlst)
    return time, LAMB, flux_matrix, flux_matrix * np.nan


def read_pycoco_template_outdir(pycoco_out_dir):
    time, fluxlst, fluxerrlst = [], [], []

    def path2t(pth):
        return float(basename(pth).split('_')[0])

    for path in sorted(glob(join(pycoco_out_dir, '*.txt')), key=path2t):
        t = path2t(path)
        lamb, flux, fluxerr = zip(np.loadtxt(path, comments='#', delimiter='\t', unpack=True))
        if isinstance(fluxerr, tuple):
            fluxerr = fluxerr[0]
        if isinstance(lamb, tuple):
            lamb = lamb[0]
        if isinstance(flux, tuple):
            flux = flux[0]

        lamb, flux, fluxerr = zip(*sorted(zip(lamb, flux, fluxerr), key=lambda tup: tup[0]))
        lamb, flux, fluxerr = np.array(lamb), np.array(flux), np.array(fluxerr)

        flux, fluxerr = spectres(LAMB, lamb, flux, fluxerr, np.nan, False)

        time.append(t)
        fluxlst.append(flux)
        fluxerrlst.append(fluxerr)

    flux_matrix = np.row_stack(fluxlst)
    fluxerr_matrix = np.row_stack(fluxerrlst)
    time = np.array(time)
    return time, LAMB, flux_matrix, fluxerr_matrix


class SN:
    def __init__(self, name=None):
        if name is None:
            return

        self.name = name
        # self.pkldtempplatepath = join(PKLD_TEMPLATES_DIR, 'pycoco_' + self.name + '.SED.pickle')
        self.features = None

        sedpath = PYCOCO_SED_PATH % self.name
        if isfile(sedpath):
            self.time, self.lamb, self.flux, self.fluxerr = read_pycoco_template_sed(sedpath)
        else:
            pycoco_out_dir = PYCOCO_FINAL_DIR % self.name
            if isdir(pycoco_out_dir):
                self.time, self.lamb, self.flux, self.fluxerr = read_pycoco_template_outdir(pycoco_out_dir)
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

    def specalbum(self):
        plotting.specalbum(self, labels=[self.name], title='0 d = ' + str(self.maxtime) + ' MJD')

    def surface(self):
        plt.contourf(self.time, self.lamb, self.flux.T, 100)
        plt.title(self.name + ', At Rest Frame After Host Corrections')
        plt.xlabel('MJD - max ' + r'$m_B$')
        plt.ylabel(lambstr)
        plt.colorbar(label=r'$f_\lambda$, scaled')
        plt.show()


def sne_list(snlistname, calcfeatures=None, sne_to_exclude=[], exclude_row_and_col=True):
    """

    Args:

        snlistname: the name of the pickle file to create/read from
        calcfeatures: function which takes snlist and edits sn.features for all sn in snlist (only used when creating)
        sne_to_exclude: (only affects display, will save with all SNe)
        exclude_row_and_col: whether to also exclude column from all feature vectors (e.g. when features is distances, only affects display)
    """

    path = join(SNLISTS_DIR, snlistname)
    if isfile(path):
        print('Reading pickled "' + snlistname + '"')
        with open(path, 'rb') as f:
            snlist_ = pickle.load(f)

    else:
        if calcfeatures is None:
            raise Exception('calcfeatures must be specified when creating new snlist')
        snlist_ = Parallel(n_jobs=NUM_JOBS, verbose=0)(delayed(SN)(nm) for nm in info_df.index.to_list())
        # snlist_ = [SN(nm) for nm in tqdm(info_df.index.to_list())]
        calcfeatures(snlist_)
        with open(path, 'wb') as f:
            pickle.dump(snlist_, f)

    exclude_idxs = [i for i, sn in enumerate(snlist_) if sn.name in sne_to_exclude]
    snlist_ = [sn for i, sn in enumerate(snlist_) if i not in exclude_idxs]
    if exclude_row_and_col:
        for sn in snlist_:
            sn.features = [x for i, x in enumerate(sn.features) if i not in exclude_idxs]

    return snlist_
