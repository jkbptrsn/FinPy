import math
import numpy as np
from scipy.stats import norm
from typing import Tuple

import models.sde as sde


class SDE(sde.SDE):
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
        """Generate path(s), at t = time, of arithmetic Brownian motion
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
        return spot + self.vol * math.sqrt(time) * realizations

    def path_grid(self,
                  spot: float,
                  time_grid: np.ndarray) -> np.ndarray:
        """Generate one path, represented on time_grid, of arithmetic
        Brownian motion using analytic expression."""
        dt = time_grid[1:] - time_grid[:-1]
        spot_moved = spot \
            + np.cumsum(self.vol * np.sqrt(dt) * norm.rvs(size=dt.shape[0]))
        return np.append(spot, spot_moved)

    def path_wise(self,
                  spot: np.ndarray,
                  time: float,
                  n_paths: int,
                  greek: str = 'delta',
                  antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths, at t = time, of arithmetic Brownian motion
        using analytic expression. The paths are used for "path-wise"
        Monte-Carlo calculation of a 'greek'.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        paths = self.path(spot, time, n_paths, antithetic)
        if greek == 'delta':
            return paths, np.ones(paths.shape[0])
        elif greek == 'vega':
            return paths, (paths - spot) / self.vol
        else:
            raise ValueError("greek can be 'delta' or 'vega'")

    def likelihood_ratio(self,
                         spot: np.ndarray,
                         time: float,
                         n_paths: int,
                         greek: str = 'delta',
                         antithetic: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths, at t = time, of arithmetic Brownian motion
        using analytic expression. The paths are used for
        'likelihood-ratio' Monte-Carlo calculation of a 'greek'.

        The density transformation theorem is used in the derivation of
        the expressions...

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        paths = self.path(spot, time, n_paths, antithetic)
        if greek == 'delta':
            wiener = (paths - spot) / self.vol
            return paths, wiener / self.vol
        elif greek == 'vega':
            normal = (paths - spot) / (self.vol * math.sqrt(time))
            return paths, (normal ** 2 - 1) / self.vol
        else:
            raise ValueError("greek can be 'delta' or 'vega'")
