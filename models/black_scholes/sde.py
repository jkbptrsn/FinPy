import math
import numpy as np
from scipy.stats import norm

import models.sde as sde


class SDE(sde.AbstractSDE):
    """Black-Scholes SDE:
    dS_t = rate * S_t * dt + vol * S_t * dW_t
    """

    def __init__(self, rate, vol):
        self._rate = rate
        self._vol = vol
        self._model_name = 'Black-Scholes'

    def __repr__(self):
        return f"{self.model_name} SDE object"

    @property
    def model_name(self):
        return self._model_name

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
        self._vol = vol_

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate realization(s) of geometric Brownian motion.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        if antithetic:
            if n_paths % 2 == 1:
                raise ValueError("In antithetic sampling, "
                                 "n_paths should be even.")
            realizations = norm.rvs(size=n_paths // 2)
            realizations = np.append(realizations, -realizations)
        else:
            realizations = norm.rvs(size=n_paths)
        return spot * np.exp((self.rate - self.vol ** 2 / 2) * time
                             + self.vol * math.sqrt(time) * realizations)
