import math
import numpy as np
from scipy.stats import norm
from typing import Tuple

import models.sde as sde


class SDE(sde.SDE):
    """Black-Scholes SDE:
    dS_t / S_t = (rate - dividend) * dt + vol * dW_t
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 dividend: float = 0):
        self._rate = rate
        self._vol = vol
        self._dividend = dividend
        self._model_name = 'Black-Scholes'

    def __repr__(self) -> str:
        return f"{self._model_name} SDE object"

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, rate_):
        self._rate = rate_

    @property
    def vol(self) -> float:
        return self._vol

    @vol.setter
    def vol(self, vol_):
        self._vol = vol_

    @property
    def dividend(self) -> float:
        return self._dividend

    @dividend.setter
    def dividend(self, dividend_):
        self._dividend = dividend_

    @property
    def model_name(self) -> str:
        return self._model_name

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate path(s), at t = time, of geometric Brownian motion
        using analytic expression.

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

    def path_time_grid(self,
                       spot: float,
                       time_grid: np.ndarray) -> np.ndarray:
        """Generate one path, represented on time_grid, of geometric
        Brownian motion using analytic expression.
        """
        dt = time_grid[1:] - time_grid[:-1]
        spot_moved = spot * np.cumprod(
            np.exp((self.rate - self.vol ** 2 / 2) * dt
                   + self.vol * np.sqrt(dt) * norm.rvs(size=dt.shape[0])))
        return np.append(spot, spot_moved)

    def path_wise(self,
                  spot: np.ndarray,
                  time: float,
                  n_paths: int,
                  greek: str = 'delta',
                  antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths, at t = time, of geometric Brownian motion
        using analytic expression. The paths are used for "path-wise"
        Monte-Carlo calculation of a 'greek'.
        Todo: See 'Estimating the greeks' lecture notes by Martin Haugh (2017)

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        paths = self.path(spot, time, n_paths, antithetic)
        if greek == 'delta':
            return paths, paths / spot
        elif greek == 'vega':
            wiener = (np.log(paths / spot)
                      - (self.rate - 0.5 * self.vol ** 2) * time) / self.vol
            return paths, paths * (wiener - self.vol * time)
        else:
            raise ValueError("greek can be 'delta' or 'vega'")

    def likelihood_ratio(self,
                         spot: np.ndarray,
                         time: float,
                         n_paths: int,
                         greek: str = 'delta',
                         antithetic: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths, at t = time, of geometric Brownian motion
        using analytic expression. The paths are used for
        'likelihood-ratio' Monte-Carlo calculation of a 'greek'.

        The density transformation theorem is used in the derivation of
        the expressions...
        Todo: See 'Estimating the greeks' lecture notes by Martin Haugh (2017)

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        paths = self.path(spot, time, n_paths, antithetic)
        if greek == 'delta':
            wiener = (np.log(paths / spot)
                      - (self.rate - 0.5 * self.vol ** 2) * time) / self.vol
            # Todo: Should wiener be divided by (self.expiry - time)?
            return paths, wiener / (spot * self.vol)
        elif greek == 'vega':
            normal = (np.log(paths / spot)
                      - (self.rate - 0.5 * self.vol ** 2) * time) \
                     / (self.vol * math.sqrt(time))
            return paths, normal ** 2 / self.vol \
                - normal * math.sqrt(time) - 1 / self.vol
        else:
            raise ValueError("greek can be 'delta' or 'vega'")
