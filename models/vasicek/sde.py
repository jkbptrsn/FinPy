import numpy as np

import models.sde as sde


class SDE(sde.SDE):
    """Vasicek SDE:
    dr_t = kappa * ( theta - r_t) * dt + vol * dW_t
    """

    def __init__(self, kappa, mean_rate, vol):
        self._kappa = kappa
        self._mean_rate = mean_rate
        self._vol = vol
        self._model_name = "Vasicek"

    def __repr__(self):
        return f"{self._model_name} SDE object"

    @property
    def model_name(self):
        return self._model_name

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, kappa_):
        self._kappa = kappa_

    @property
    def mean_rate(self):
        return self._mean_rate

    @mean_rate.setter
    def mean_rate(self, mean_rate_):
        self._mean_rate = mean_rate_

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
        """Generate paths(s), at t = time, of Ornstein-Uhlenbeck motion
        using analytic expression.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        pass

    def path_grid(self,
                  spot: float,
                  time_grid: np.ndarray) -> np.ndarray:
        """Generate one path, represented on time_grid, of
        Ornstein-Uhlenbeck motion using analytic expression.
        """
        pass
