import pickle
from os.path import isdir, isfile

from joblib import Parallel, delayed

from utils import *

"""input:"""

timeclipdict = {'SN2006aj': (2.5, np.inf),
                'slimSN2011bm': (-np.inf, 55680)}  # from the original time count, that in the pycoco output
timeclip_since = (0, 51)  # time count here is time since max
LAMB = np.arange(4000, 8000 + 40, 40)  # AA, grid for all SNe
TIME = np.arange(*timeclip_since, 1)  # days since max, grid for all SNe


class SN:
    """
    contains the (final, interpolated) data of a single SN
    """

    def __init__(self, name=None):
        if name is None:
            return

        self.internal_name = name
        self.name = name if name not in iaunames else iaunames[name]
        self.type = info_df['FullType'][name] if not pd.isna(info_df['FullType'][name]) else info_df['Type'][name]

        # by reading the .SED files:
        # sedpath = PYCOCO_SED_PATH % self.name
        # if isfile(sedpath):
        #     self.time, self.lamb, self.flux, self.fluxerr = read_pycoco_template_sed(sedpath, LAMB)
        # else:
        pycoco_out_dir = PYCOCO_FINAL_DIR % name
        if isdir(pycoco_out_dir):
            self.time, self.lamb, self.flux, self.fluxerr = read_pycoco_template_outdir(pycoco_out_dir, LAMB)
        else:
            raise Exception('no template found for %s' % self.name)

        if self.name in timeclipdict:
            timeclip = timeclipdict[self.name]
            keep = (self.time >= timeclip[0]) * (self.time <= timeclip[1])
            self.time, self.flux, self.fluxerr = self.time[keep], self.flux[keep, :], self.fluxerr[keep, :]

        # since maximum:
        # self.maxtime = timesincemax(self.time, self.lamb, self.flux, self.fluxerr)
        # self.time = self.time - self.maxtime
        # since explosion:
        self.exptime = min(self.time)
        self.time = self.time - self.exptime

        self.flux /= np.nanmax(self.flux)

        if timeclip_since is not None:
            keep = (self.time >= timeclip_since[0]) * (self.time < timeclip_since[1])
            self.time, self.flux, self.fluxerr = self.time[keep], self.flux[keep, :], self.fluxerr[keep, :]

        self.iflux = interp1d(self.time, self.flux, axis=0, bounds_error=False, fill_value=np.nan)
        self.iflux = self.iflux(TIME)

    def specalbum(self):
        specalbum(self, labels=[self.name], title='0 d = ' + 'explosion')

    def surface(self):
        plt.contourf(self.time, self.lamb, self.flux.T, 100)
        plt.title(self.name + ', At Rest Frame After Host Corrections')
        plt.xlabel('days since explosion')
        plt.ylabel(r'wavelength [$\AA$]')
        plt.colorbar(label='Flux rescaled')
        plt.show()


def calcfeatures(snlist):
    """
    a function which takes snlist and returns some matrix X of features
    """
    X = []
    for sn in snlist:
        ftilde = np.gradient(np.log10(sn.flux), axis=1)
        # ftilde = sn.flux
        iflux = interp1d(sn.time, ftilde, axis=0, bounds_error=False, fill_value=np.nan)
        iflux = iflux(TIME)
        X.append(iflux.flatten())

    X = np.row_stack(X)
    return X


def sne_list(sne_to_exclude=None, exclude_row_and_col=False):
    """
    The pickled_snlist is read if it exists. Otherwise, it is created. The list of SNe is returned.

    Args:
        sne_to_exclude: name list, only takes effect when reading the pickled snlist
        exclude_row_and_col: should be True when e.g. features are a distance matrix
    """

    if isfile(SNLIST_PATH):
        if sne_to_exclude is None or exclude_row_and_col is None:
            raise Exception('must enter sne_to_exclude (name list) and set exclude_row_and_col (bool)')
        print('=== Reading pickled snlist ===')
        with open(SNLIST_PATH, 'rb') as f:
            snlist_, X = pickle.load(f)

    else:
        snlist_ = Parallel(n_jobs=NUM_JOBS, verbose=0)(delayed(SN)(nm) for nm in info_df.index.to_list())
        # snlist_ = [SN(nm) for nm in tqdm(info_df.index.to_list())]
        X = calcfeatures(snlist_)
        with open(SNLIST_PATH, 'wb') as f:
            pickle.dump((snlist_, X), f)

    include_idxs = [i for i, sn in enumerate(snlist_) if sn.name not in sne_to_exclude and
                    sn.internal_name not in sne_to_exclude]
    snlist_ = [snlist_[i] for i in include_idxs]
    X = X[include_idxs, :]
    if exclude_row_and_col:
        X = X[:, include_idxs]

    return snlist_, X
