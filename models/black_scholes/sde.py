import math
import typing

import numpy as np

from models import sde
from utils import global_types
from utils import misc


class SDE(sde.SDE):
    """SDE for the Black-Scholes model
        dS_t / S_t = (rate - dividend) * dt + vol * dW_t

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
        self.rate = rate
        self.vol = vol
        self.event_grid = event_grid
        self.dividend = dividend

        self.model_name = global_types.ModelName.BLACK_SCHOLES

        self.price_mean = np.zeros(self.event_grid.size)
        self.price_variance = np.zeros(self.event_grid.size)

    def __repr__(self) -> str:
        return f"{self.model_name} SDE object"

    def initialization(self):
        """Initialize the Monte-Carlo engine by calculating mean and
        variance of the stock price process.
        """
        self.calc_price_mean()
        self.calc_price_variance()

    def calc_price_mean(self):
        """Conditional mean of stock price process."""
        self.price_mean[0] = 1
        factor = self.rate - self.dividend
        self.price_mean[1:] = np.exp(factor * np.diff(self.event_grid))

    def calc_price_variance(self):
        """Conditional variance of stock price process."""
        factor = self.rate - self.dividend
        self.price_variance[1:] = \
            np.exp(2 * factor * np.diff(self.event_grid)) \
            * (np.exp(self.vol ** 2 * np.diff(self.event_grid)) - 1)

    def price_increment(self,
                        spot: (float, np.ndarray),
                        time_idx: int,
                        normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment stock price process (the spot price is subtracted
        to get the increment).
        """
        mean = spot * self.price_mean[time_idx]
        variance = spot ** 2 * self.price_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

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
        price = np.zeros((self.event_grid.size, n_paths))
        price[0] = spot
        if seed is not None:
            np.random.seed(seed)
        for time_idx in range(1, self.event_grid.size):
            realizations = \
                misc.normal_realizations(n_paths, antithetic=antithetic)
            price[time_idx] = price[time_idx - 1] \
                + self.price_increment(price[time_idx - 1], time_idx,
                                       realizations)
        return price

    def path_wise(self,
                  spot: np.ndarray,
                  time: float,
                  n_paths: int,
                  greek: str = 'delta',
                  antithetic: bool = False) \
            -> typing.Tuple[np.ndarray, np.ndarray]:
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
            -> typing.Tuple[np.ndarray, np.ndarray]:
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
