import math
import typing

import numpy as np

from utils import global_types
from utils import misc


class SDE:
    """SDE for the Black-Scholes model.

    Geometric Brownian motion is given by
        dS_t / S_t = (rate - dividend) * dt + vol * dW_t,
    which leads to
        S_t = S_0 exp((rate - dividend - vol^2 / 2) * t + vol * W_t).

    S_t is log-normally distributed
        ln(S_t) ~ N((rate - dividend - vol^2 / 2) * t, vol^2).

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.
        dividend: Continuous dividend yield. Default value is 0.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            event_grid: np.ndarray,
            dividend: float = 0):
        self.rate = rate
        self.vol = vol
        self.event_grid = event_grid
        self.dividend = dividend

        self.spot = None
        self.n_paths = None
        self.rng = None
        self.seed = None
        self.antithetic = None

        self.discount_grid = np.exp(-rate * event_grid)

        self.model = global_types.Model.BLACK_SCHOLES
        self.solution = None

    def __repr__(self) -> str:
        return f"{self.model} SDE object"

    def initialization(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False):
        """Initialization of path generation."""
        self.spot = spot
        self.n_paths = n_paths
        self.rng = rng
        self.seed = seed
        self.antithetic = antithetic

    def _increment(
            self,
            spot: typing.Union[float, np.ndarray],
            time_idx: int,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment stock price process.

        Spot price is subtracted to get the increment.
        """
        dt = self.event_grid[time_idx] - self.event_grid[time_idx - 1]
        mean = (self.rate - self.dividend - self.vol ** 2 / 2) * dt
        std = self.vol * math.sqrt(dt)
        return spot * np.exp(mean + std * normal_rand) - spot

    def paths(self):
        """Monte-Carlo paths using exact discretization."""
        grid = np.zeros((self.event_grid.size, self.n_paths))
        grid[0, :] = self.spot
        if self.rng:
            rng = self.rng
        else:
            rng = np.random.default_rng(self.seed)
        for idx in range(1, self.event_grid.size):
            realizations = \
                misc.normal_realizations(self.n_paths, rng, self.antithetic)
            grid[idx, :] = grid[idx - 1, :] \
                + self._increment(grid[idx - 1, :], idx, realizations)
        self.solution = grid

    def price(
            self,
            instrument,
            time_idx: int) -> (float, float):
        """Price of instrument."""
        payoff = instrument.payoff(self.solution[time_idx, :])
        discount = math.exp(-self.rate * self.event_grid[time_idx])
        price = discount * payoff
        return price.mean(), price.std(), misc.monte_carlo_error(price)
