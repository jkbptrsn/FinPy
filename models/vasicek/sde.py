import math
import numpy as np
from scipy.stats import norm
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types


class SDE(sde.SDE):
    """Vasicek SDE for the short rate
        dr_t = kappa * (mean_rate - r_t) * dt + vol * dW_t.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float):
        self._kappa = kappa
        self._mean_rate = mean_rate
        self._vol = vol
        self._model_name = global_types.ModelName.VASICEK

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
    def model_name(self) -> global_types.ModelName:
        return self._model_name

    def rate_mean(self,
                  spot: (float, np.ndarray),
                  dt: float) -> (float, np.ndarray):
        """Conditional mean of short rate process.
        Eq. (10.12), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        exp_kappa = math.exp(- self._kappa * dt)
        return spot * exp_kappa + self._mean_rate * (1 - exp_kappa)

    def rate_variance(self,
                      dt: float) -> float:
        """Conditional variance of short rate process.
        Eq. (10.13), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        two_kappa = 2 * self._kappa
        exp_two_kappa = math.exp(- two_kappa * dt)
        return self._vol ** 2 * (1 - exp_two_kappa) / two_kappa

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       dt: float,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment short rate process (the spot rate is subtracted to
        get the increment).
        """
        return self.rate_mean(spot, dt) \
            + math.sqrt(self.rate_variance(dt)) * normal_rand - spot

    def discount_mean(self,
                      spot: (float, np.ndarray),
                      dt: float) -> (float, np.ndarray):
        """Conditional mean of discount process, i.e.,
        - int_t^{t+dt} r_u du.
        Eq. (10.12+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        exp_kappa = math.exp(- self._kappa * dt)
        return - self._mean_rate * dt \
            - (spot - self._mean_rate) * (1 - exp_kappa) / self._kappa

    def discount_variance(self,
                          dt: float) -> float:
        """Conditional variance of discount process, i.e.,
        - int_t^{t+dt} r_u du.
        Eq. (10.13+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        vol_sq = self._vol ** 2
        exp_kappa = math.exp(- self._kappa * dt)
        two_kappa = 2 * self._kappa
        exp_two_kappa = math.exp(- two_kappa * dt)
        kappa_cubed = self._kappa ** 3
        return vol_sq * (4 * exp_kappa - 3 + two_kappa * dt
                         - exp_two_kappa) / (2 * kappa_cubed)

    def discount_increment(self,
                           spot: (float, np.ndarray),
                           time: float,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment discount process."""
        return self.discount_mean(spot, time) \
            + np.sqrt(self.discount_variance(time)) * normal_rand

    def covariance(self,
                   dt: float) -> float:
        """Covariance between between short rate and discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        vol_sq = self._vol ** 2
        kappa_sq = self._kappa ** 2
        exp_kappa = math.exp(- self._kappa * dt)
        exp_two_kappa = math.exp(- 2 * self._kappa * dt)
        return vol_sq * (2 * exp_kappa - exp_two_kappa - 1) / (2 * kappa_sq)

    def correlation(self,
                    dt: float) -> float:
        """Correlation between between short rate and discount
        processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        covariance = self.covariance(dt)
        rate_var = self.rate_variance(dt)
        discount_var = self.discount_variance(dt)
        return covariance / math.sqrt(rate_var * discount_var)

    def cholesky(self,
                 dt: float,
                 n_paths: int) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):
        """Correlated standard normal random variables using Cholesky
        decomposition. In the 2-D case, the transformation is simply:
        x1, correlation * x1 + sqrt(1 - correlation ** 2) * x2
        """
        correlation = self.correlation(dt)
        corr_matrix = np.array([[1, correlation], [correlation, 1]])
        corr_matrix = np.linalg.cholesky(corr_matrix)
        x1 = norm.rvs(size=n_paths)
        x2 = norm.rvs(size=n_paths)
        return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
            corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate paths(s), at t = time, of correlated short rate and
        discount factor using Exact discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        if antithetic:
            if n_paths % 2 == 1:
                raise ValueError("In antithetic sampling, "
                                 "n_paths should be even.")
            rate, discount = self.cholesky(time, n_paths // 2)
            rate = np.append(rate, -rate)
            discount = np.append(discount, -discount)
        else:
            rate, discount = self.cholesky(time, n_paths)
        return self.rate_increment(spot, time, rate), \
            self.discount_increment(spot, time, discount)

    def path_time_grid(self,
                       spot: float,
                       time_grid: np.ndarray) -> np.ndarray:
        """Generate one path, represented on time_grid, of
        Ornstein-Uhlenbeck motion using analytic expression.

        returns tuple of np.ndarrays...
        """
        delta_t = time_grid[1:] - time_grid[:-1]
        rate = np.zeros(delta_t.shape[0] + 1)
        rate[0] = spot
        discount = np.zeros(delta_t.shape[0] + 1)
        for count, dt in enumerate(delta_t):
            x_rate, x_discount = self.cholesky(dt, 1)
            rate[count + 1] = rate[count] \
                + self.rate_increment(rate[count], dt, x_rate)
            discount[count + 1] = discount[count] \
                + self.discount_increment(rate[count], dt, x_discount)
        return rate, np.exp(discount)
