import math
import numpy as np
from scipy.stats import norm

import models.sde as sde


class SDE(sde.AbstractSDE):
    """Bachelier SDE:
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

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate realizations of arithmetic Brownian motion.

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
        return spot + self.vol * math.sqrt(time) * realizations
