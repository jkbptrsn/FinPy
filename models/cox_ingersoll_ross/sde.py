import math
import numpy as np
from typing import Tuple

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
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.event_grid = event_grid

        self.model_name = global_types.Model.CIR

    def __repr__(self):
        return f"{self.model_name} SDE object"

    def rate_mean(self,
                  spot: (float, np.ndarray),
                  delta_t: float) -> (float, np.ndarray):
        """Conditional mean of short rate process.
        Eq. (3.23), Brigo & Mercurio 2007.
        """
        exp_kappa = math.exp(- self.kappa * delta_t)
        return spot * exp_kappa + self.mean_rate * (1 - exp_kappa)

    def rate_variance(self,
                      spot: float,
                      delta_t: (float, np.ndarray)) -> (float, np.ndarray):
        """Conditional variance of short rate process.
        Eq. (3.23), Brigo & Mercurio 2007.
        """
        vol_sq = self.vol ** 2
        two_kappa = 2 * self.kappa
        exp_kappa = np.exp(- self.kappa * delta_t)
        exp_two_kappa = np.exp(- two_kappa * delta_t)
        return spot * vol_sq * (exp_kappa - exp_two_kappa) / self.kappa \
            + self.mean_rate * vol_sq * (1 - exp_kappa) ** 2 / two_kappa

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
