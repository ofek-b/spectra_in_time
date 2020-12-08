from abc import ABC, abstractmethod

from scipy.interpolate import interp1d
from scipy.optimize import minimize

import plotting
import numpy as np


class BaseDissimilarity(ABC):
    def __init__(self, tr, tp):
        assert np.all(tr.lamb == tp.lamb)
        bounds, x0, nrmlz = self.bounds_x0_nrmlz()

        self.tr, self.tp = tr, tp
        self.tp_flux_func = interp1d(self.tp.time, self.tp.flux / nrmlz(self.tp.flux), axis=0, fill_value=np.nan,
                                     bounds_error=False)
        self.tr_flux_func = interp1d(self.tr.time, self.tr.flux / nrmlz(self.tr.flux), axis=0, fill_value=np.nan,
                                     bounds_error=False)

        self.minres = minimize(self.objective, bounds=bounds, x0=np.array(x0))
        self.result = self.minres.fun

        if np.isnan(self.result):
            raise Exception(tr.name + ', ' + tp.name)

    @abstractmethod
    def bounds_x0_nrmlz(self):
        """
        Returns the bounds and initial point for optimization
        """
        pass

    @abstractmethod
    def objective(self, x):
        """
        Minimization objective function
        """
        pass

    @abstractmethod
    def winners(self):
        """
        This function returns the speclists of tr, tp at optimum (self.minres.x)
        """
        pass

    def plot(self):
        title = 'RMS' + r'$\Delta f$ = ' + "{:.2E}".format(self.result)
        plotting.specalbum(list(self.winners()), labels=[self.tp.name, self.tr.name + ' (fitted)'], title=title)
