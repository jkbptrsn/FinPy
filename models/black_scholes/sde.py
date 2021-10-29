import math
import numpy as np
from scipy.stats import norm


class SDE:
    """
    Black-Scholes SDE: dS_t = rate * S_t * dt + vol * S_t * dW_t
    """

    def __init__(self, rate, vol):
        self._rate = rate
        self._vol = vol
        self._model_name = 'Black-Scholes'

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate_):
        self._rate = rate_

    @property
    def vol(self):
        return self._vol

    @vol.setter
    def vol(self, vol_):
        self._rate = vol_

    @property
    def model_name(self):
        return self._model_name

    def path(self, spot, time, n_paths, antithetic=False):
        """
        Generate realizations of Geometric Brownian motion

        todo: What if len(spot) != 1 and n_paths != 1, and len(spot) != n_paths?
        todo: What if n_paths is odd?

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float
        n_paths : int
        antithetic : bool
            Antithetic sampling for Monte-Carlo variance reduction

        Returns
        -------
        numpy.ndarray
        """
        if antithetic:
            realizations = norm.rvs(size=n_paths // 2)
            realizations = np.append(realizations, -realizations)
        else:
            realizations = norm.rvs(size=n_paths)
        return spot * np.exp((self.rate - self.vol ** 2 / 2) * time
                             + self.vol * math.sqrt(time) * realizations)
