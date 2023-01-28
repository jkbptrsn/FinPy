import math
import numpy as np

import models.sde as sde
import utils.global_types as global_types
import utils.misc as misc


class SDE(sde.SDE):
    """SDE for the Bachelier model
        dS_t = rate * dt + vol * dW_t

    - rate: Risk-free interest rate
    - vol: Volatility
    - event_grid: Event dates, i.e., trade date, payment dates, etc.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray):
        self.rate = rate
        self.vol = vol
        self.event_grid = event_grid

        self.model_name = global_types.ModelName.BACHELIER

        self._price_mean = np.zeros(self.event_grid.size)
        self._price_variance = np.zeros(self.event_grid.size)

    def __repr__(self):
        return f"{self.model_name} SDE object"

    def initialization(self):
        """Initialize the Monte-Carlo engine by calculating mean and
        variance of the stock price process.
        """
        self.price_mean()
        self.price_variance()

    def price_mean(self):
        """Conditional mean of stock price process."""
        self._price_mean[1:] = self.rate * np.diff(self.event_grid)

    def price_variance(self):
        """Conditional variance of stock price process."""
        self._price_variance[1:] = self.vol ** 2 * np.diff(self.event_grid)

    def price_increment(self,
                        time_idx: int,
                        normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment stock price process."""
        mean = self._price_mean[time_idx]
        variance = self._price_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

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

        rng = np.random.default_rng(seed)

        for time_idx in range(1, self.event_grid.size):
            realizations = \
                misc.normal_realizations(n_paths, rng, antithetic=antithetic)
            price[time_idx] = price[time_idx - 1] \
                + self.price_increment(time_idx, realizations)
        return price
