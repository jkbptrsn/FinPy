import math
import numpy as np
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types
import utils.misc as misc


class SDE(sde.SDE):
    """SDE for the short rate in the Vasicek model
        dr_t = kappa * (mean_rate - r_t) * dt + vol * dW_t

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

        self._model_name = global_types.ModelName.VASICEK

        self._rate_mean = np.zeros((self._event_grid.size, 2))
        self._rate_variance = np.zeros(self._event_grid.size)
        self._discount_mean = np.zeros((self._event_grid.size, 2))
        self._discount_variance = np.zeros(self._event_grid.size)
        self._covariance = np.zeros(self._event_grid.size)

    def __repr__(self):
        return f"{self._model_name} SDE object"

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self,
              kappa_: float):
        self._kappa = kappa_

    @property
    def mean_rate(self) -> float:
        return self._mean_rate

    @mean_rate.setter
    def mean_rate(self,
                  mean_rate_: float):
        self._mean_rate = mean_rate_

    @property
    def vol(self) -> float:
        return self._vol

    @vol.setter
    def vol(self,
            vol_: float):
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

    def initialization(self):
        """Initialize the Monte-Carlo engine by calculating mean and
        variance of the short rate and discount processes, respectively.
        """
        self.rate_mean()
        self.rate_variance()
        self.discount_mean()
        self.discount_variance()
        self.covariance()

    def rate_mean(self):
        """Conditional mean of short rate process.
        Eq. (10.12), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        exp_kappa = np.exp(-self._kappa * np.diff(self._event_grid))
        self._rate_mean[0, 0] = 1
        self._rate_mean[1:, 0] = exp_kappa
        self._rate_mean[1:, 1] = self._mean_rate * (1 - exp_kappa)

    def rate_variance(self):
        """Conditional variance of short rate process.
        Eq. (10.13), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        two_kappa = 2 * self._kappa
        exp_two_kappa = np.exp(-two_kappa * np.diff(self._event_grid))
        self._rate_variance[1:] = \
            self._vol ** 2 * (1 - exp_two_kappa) / two_kappa

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       time_idx: int,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment short rate process (the spot rate is subtracted to
        get the increment).
        """
        mean = \
            self._rate_mean[time_idx, 0] * spot + self._rate_mean[time_idx, 1]
        variance = self._rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def discount_mean(self):
        """Conditional mean of discount process, i.e.,
        -int_t^{t+dt} r_u du.
        Eq. (10.12+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dt = np.diff(self._event_grid)
        exp_kappa = np.exp(-self._kappa * dt)
        exp_kappa = (1 - exp_kappa) / self._kappa
        self._discount_mean[1:, 0] = -exp_kappa
        self._discount_mean[1:, 1] = self._mean_rate * (exp_kappa - dt)

    def discount_variance(self):
        """Conditional variance of discount process, i.e.,
        -int_t^{t+dt} r_u du.
        Eq. (10.13+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dt = np.diff(self._event_grid)
        vol_sq = self._vol ** 2
        exp_kappa = np.exp(-self._kappa * dt)
        two_kappa = 2 * self._kappa
        exp_two_kappa = np.exp(-two_kappa * dt)
        kappa_cubed = self._kappa ** 3
        self._discount_variance[1:] = \
            vol_sq * (4 * exp_kappa - 3 + two_kappa * dt
                      - exp_two_kappa) / (2 * kappa_cubed)

    def discount_increment(self,
                           rate_spot: (float, np.ndarray),
                           time_idx: int,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment discount process."""
        mean = self._discount_mean[time_idx, 0] * rate_spot \
            + self._discount_mean[time_idx, 1]
        variance = self._discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def covariance(self):
        """Covariance between between short rate and discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dt = np.diff(self._event_grid)
        vol_sq = self._vol ** 2
        kappa_sq = self._kappa ** 2
        exp_kappa = np.exp(-self._kappa * dt)
        exp_two_kappa = np.exp(-2 * self._kappa * dt)
        self._covariance[1:] = \
            vol_sq * (2 * exp_kappa - exp_two_kappa - 1) / (2 * kappa_sq)

    def correlation(self,
                    time_idx: int) -> float:
        """Correlation between between short rate and discount
        processes.
        """
        covariance = self._covariance[time_idx]
        rate_var = self._rate_variance[time_idx]
        discount_var = self._discount_variance[time_idx]
        return covariance / math.sqrt(rate_var * discount_var)

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
        rate = np.zeros((self._event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self._event_grid.size, n_paths))
        for time_idx in range(1, self._event_grid.size):
            correlation = self.correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, seed, antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self.rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self.discount_increment(rate[time_idx - 1], time_idx,
                                          x_discount)
        # Get discount factors at event dates
        discount = np.exp(discount)
        return rate, discount
