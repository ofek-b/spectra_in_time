import numpy as np

from dissimilarity.basedissimilarity import BaseDissimilarity
from pycoco2SNobject import SN


class Dissimilarity(BaseDissimilarity):
    def __init__(self, tr, tp):
        super().__init__(tr, tp)

    def objective(self, x):
        timeshift, fluxscale,fluxshift = x
        iflux_tp = self.tp_flux_func(self.tr.time + timeshift)

        dif = fluxscale * self.tr.flux + fluxshift - iflux_tp
        rms = np.sqrt(np.nanmean(dif ** 2))
        return rms

    def winners(self):
        timeshift, fluxscale,fluxshift = self.minres.x
        _, _, nrmlz = self.bounds_x0_nrmlz()
        iflux_tp = self.tp_flux_func(self.tr.time + timeshift)
        iflux_tr = self.tr.flux / nrmlz(self.tp.flux)

        tp_w = SN()
        tp_w.time = self.tr.time + timeshift
        tp_w.flux = iflux_tp
        tp_w.lamb = self.tp.lamb

        tr_w = SN()
        tr_w.time = self.tr.time
        tr_w.flux = fluxscale * iflux_tr
        tr_w.lamb = self.tr.lamb

        return tp_w, tr_w

    def bounds_x0_nrmlz(self):
        return [(-5,5), (0.9, 1.1), (-0.05,0.05)], [0, 1,0], np.nanmax
