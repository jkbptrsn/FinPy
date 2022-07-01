import math
import numpy as np
from scipy.stats import norm

import models.sde as sde
import utils.global_types as global_types


class SDE(sde.SDE):
    """SDE for the Cox-Ingersoll-Ross (CIR) model
        dr_t = kappa * ( mean_rate - r_t) * dt + vol * sqrt(r_t) * dW_t
    subject to the condition
        2 * kappa * theta > vol ** 2
    such that r_t = 0 is precluded.

    - kappa: Speed of mean reversion
    - mean_rate: Long-time mean
    - vol: Volatility
    - event_grid: event dates, i.e., trade date, payment dates, etc.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray):
        self._kappa = kappa
        self._mean_rate = mean_rate
        self._vol = vol
        self._event_grid = event_grid

        self._model_name = global_types.ModelName.CIR

    def __repr__(self):
        return f"{self._model_name} SDE object"

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, kappa_):
        self._kappa = kappa_

    @property
    def mean_rate(self) -> float:
        return self._mean_rate

    @mean_rate.setter
    def mean_rate(self, mean_rate_):
        self._mean_rate = mean_rate_

    @property
    def vol(self) -> float:
        return self._vol

    @vol.setter
    def vol(self, vol_):
        self._vol = vol_

    @property
    def event_grid(self) -> np.ndarray:
        return self._event_grid

    @event_grid.setter
    def event_grid(self,
                   event_grid_: np.ndarray):
        self._event_grid = event_grid_

    @property
    def model_name(self) -> global_types.ModelName:
        return self._model_name

    def rate_mean(self,
                  spot: (float, np.ndarray),
                  delta_t: float) -> (float, np.ndarray):
        """Conditional mean of short rate process.
        Eq. (3.23), Brigo & Mercurio 2007.
        """
        exp_kappa = math.exp(- self._kappa * delta_t)
        return spot * exp_kappa + self._mean_rate * (1 - exp_kappa)

    def rate_variance(self,
                      spot: float,
                      delta_t: (float, np.ndarray)) -> (float, np.ndarray):
        """Conditional variance of short rate process.
        Eq. (3.23), Brigo & Mercurio 2007.
        """
        vol_sq = self._vol ** 2
        two_kappa = 2 * self._kappa
        exp_kappa = np.exp(- self._kappa * delta_t)
        exp_two_kappa = np.exp(- two_kappa * delta_t)
        return spot * vol_sq * (exp_kappa - exp_two_kappa) / self._kappa \
            + self._mean_rate * vol_sq * (1 - exp_kappa) ** 2 / two_kappa

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        pass

    def path_time_grid(self,
                       spot: float,
                       time_grid: np.ndarray) -> np.ndarray:
        """Generate one path, represented on time_grid, of
        Ornstein-Uhlenbeck motion using analytic expression.
        """
        pass

    def paths(self,
              spot: float,
              n_paths: int,
              seed: int = None,
              antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths represented on _event_grid of correlated short
        rate and discount processes using exact discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        pass
