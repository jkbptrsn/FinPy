import math
import numpy as np
from scipy.stats import norm

import models.sde as sde


class SDE(sde.AbstractSDE):
    """
    Bachelier SDE class
    dS_t = vol * dW_t
    todo: Extend to dS_t = rate * S_t dt + vol * dW_t
    todo: https://quant.stackexchange.com/questions/32863/bachelier-model-call-option-pricing-formula
    """
    def __init__(self, vol):
        self._vol = vol
        self._model_name = 'Bachelier'

    def __repr__(self):
        return f"{self.model_name} SDE object"

    @property
    def model_name(self):
        return self._model_name

    @property
    def vol(self):
        return self._vol

    @vol.setter
    def vol(self, vol_):
        self._vol = vol_

    def path(self, spot, time, n_paths, antithetic=False):
        """
        Generate realizations of arithmetic Brownian motion

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float
        n_paths : int
        antithetic : bool
            Antithetic sampling for Monte-Carlo variance reduction

        Returns
        -------
        float / numpy.ndarray
        """
        if antithetic:
            if n_paths % 2 == 1:
                raise ValueError("Antithetic sampling: n_paths is odd")
            realizations = norm.rvs(size=n_paths // 2)
            realizations = np.append(realizations, -realizations)
        else:
            realizations = norm.rvs(size=n_paths)
        if (len(spot) > 1 and n_paths > 1) and (len(spot) != n_paths):
            raise ValueError("len(spot) != n_paths")
        return spot + self.vol * math.sqrt(time) * realizations
