import math
import numpy as np
from scipy.stats import norm
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types
import utils.misc as misc


class SDE(sde.SDE):
    """SDE for the Bachelier model
        dS_t = (rate - dividend) * dt + vol * dW_t

    - rate: Risk-free interest rate
    - vol: Volatility
    - event_grid: Event dates, i.e., trade date, payment dates, etc.
    - dividend: Dividend yield
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 dividend: float = 0):
        self._rate = rate
        self._vol = vol
        self._event_grid = event_grid
        self._dividend = dividend

        self._model_name = global_types.ModelName.BACHELIER

        self._price_mean = np.zeros(self._event_grid.size)
        self._price_variance = np.zeros(self._event_grid.size)

    def __repr__(self):
        return f"{self._model_name} SDE object"

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

    @property
    def event_grid(self):
        return self._event_grid

    @event_grid.setter
    def event_grid(self,
                   event_grid_: np.ndarray):
        self._event_grid = event_grid_

    @property
    def dividend(self) -> float:
        return self._dividend

    @dividend.setter
    def dividend(self,
                 dividend_: float):
        self._dividend = dividend_

    @property
    def model_name(self) -> str:
        return self._model_name

    def price_mean(self):
        """Conditional mean of stock price process."""
        self._price_mean[1:] = \
            (self._rate - self._dividend) * np.diff(self._event_grid)

    def price_variance(self):
        """Conditional variance of stock price process."""
        self._price_variance[1:] = self._vol ** 2 * np.diff(self._event_grid)

    def price_increment(self,
                        time_idx: int,
                        normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment stock price process."""
        mean = self._price_mean[time_idx]
        variance = self._price_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate paths(s), at t = time, of geometric Brownian motion
        using analytic expression.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        pass

    def path_grid(self,
                  spot: float,
                  time_grid: np.ndarray) -> np.ndarray:
        """Generate one path, represented on time_grid, of geometric
        Brownian motion using analytic expression.
        """
        pass

    def paths(self,
              spot: float,
              n_paths: int,
              seed: int = None,
              antithetic: bool = False) -> np.ndarray:
        """Generate paths represented on _event_grid of equity price
        process using exact discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        price = np.zeros((self._event_grid.size, n_paths))
        price[0] = spot
        for time_idx in range(1, self._event_grid.size):
            realizations = misc.normal_realizations(n_paths, seed, antithetic)
            price[time_idx] = price[time_idx - 1] \
                + self.price_increment(time_idx, realizations)
        return price
