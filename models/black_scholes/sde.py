import abc
import math
import typing

import numpy as np

from utils import global_types
from utils import misc


class Sde(metaclass=abc.ABCMeta):
    """SDE for stock price process in Black-Scholes model.

    The price process is defined by
        dS_t / S_t = (rate - dividend) * dt + vol * dW_t,
    where vol denotes the volatility. W_t is a Brownian motion process
    under the risk-neutral measure Q.

    Geometric Brownian motion is given by
        S_t = S_0 exp((rate - dividend - vol^2 / 2) * t + vol * W_t),
    thus S_t is log-normally distributed
        ln(S_t) ~ N((rate - dividend - vol^2 / 2) * t, vol^2 * t).

    See Hull (2015), Chapter 15.

    Attributes:
        rate: Risk-free interest rate.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
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

        self.model = global_types.Model.BLACK_SCHOLES

    @abc.abstractmethod
    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Generation of Monte-Carlo paths.

        Args:
            spot: Stock price at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        pass


class SdeExact(Sde):
    """SDE for stock price process in Black-Scholes model.

    The price process is defined by
        dS_t / S_t = (rate - dividend) * dt + vol * dW_t,
    where vol denotes the volatility. W_t is a Brownian motion process
    under the risk-neutral measure Q.

    See Hull (2015), Chapter 15.

    Monte-Carlo paths constructed using exact discretization.

    Attributes:
        rate: Risk-free interest rate.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            event_grid: np.ndarray,
            dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)

        self.discount_grid = np.exp(-self.rate * self.event_grid)

        # Arrays used for discretization.
        self.price_mean = np.zeros(self.event_grid.size)
        self.price_variance = np.zeros(self.event_grid.size)

        self.price_paths = None
        self.mc_estimate = None
        self.mc_error = None

        self.initialization()

    def initialization(self) -> None:
        """Initialization of Monte-Carlo solver."""
        self._calc_price_mean()
        self._calc_price_variance()

    def _calc_price_mean(self) -> None:
        """Conditional mean of stock price process.

        See Hull (2015), Chapter 15.
        """
        dt = np.diff(self.event_grid)
        self.price_mean[1:] = (
                (self.rate - self.dividend - self.vol ** 2 / 2) * dt)

    def _calc_price_variance(self) -> None:
        """Conditional variance of stock price process.

        See Hull (2015), Chapter 15.
        """
        dt = np.diff(self.event_grid)
        self.price_variance[1:] = self.vol ** 2 * dt

    def _price_increment(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment stock price process one time step.

        The spot stock price is subtracted to get the increment.

        Args:
            spot: Stock price at event corresponding to event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Increment of stock price process.
        """
        mean = self.price_mean[event_idx]
        variance = self.price_variance[event_idx]
        return spot * np.exp(mean + math.sqrt(variance) * normal_rand) - spot

    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False):
        """Generation of Monte-Carlo paths using exact discretization.

        Args:
            spot: Stock price at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        # Paths of stock price process.
        paths = np.zeros((self.event_grid.size, n_paths))
        paths[0] = spot
        for event_idx in range(1, self.event_grid.size):
            # Realizations of standard normal random variables.
            x_price = misc.normal_realizations(n_paths, rng, antithetic)
            # Increment of stock price process, and update.
            increment = self._price_increment(paths[event_idx - 1], event_idx,
                                              x_price)
            paths[event_idx] = paths[event_idx - 1] + increment
        # Update.
        self.price_paths = paths


class SdeEuler(Sde):
    """SDE for stock price process in Black-Scholes model.

    The price process is defined by
        dS_t / S_t = (rate - dividend) * dt + vol * dW_t,
    where vol denotes the volatility. W_t is a Brownian motion process
    under the risk-neutral measure Q.

    See Hull (2015), Chapter 15.

    Monte-Carlo paths constructed using Euler-Maruyama discretization.

    Attributes:
        rate: Risk-free interest rate.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            event_grid: np.ndarray,
            dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)

        self.discount_grid = np.exp(-self.rate * self.event_grid)

        self.price_paths = None
        self.mc_estimate = None
        self.mc_error = None

    def _price_increment(
            self,
            spot: typing.Union[float, np.ndarray],
            dt: float,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment stock price process one time step.

        The spot stock price is subtracted to get the increment.

        Args:
            spot: Stock price at event corresponding to event_idx - 1.
            dt: Time step.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Increment of stock price process.
        """
        return (self.rate - self.dividend) * spot * dt \
            + self.vol * spot * math.sqrt(dt) * normal_rand

    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False):
        """Generation of Monte-Carlo paths using Euler discretization.

        Args:
            spot: Stock price at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        # Paths of stock price process.
        paths = np.zeros((self.event_grid.size, n_paths))
        paths[0] = spot
        for idx, dt in enumerate(np.diff(self.event_grid)):
            event_idx = idx + 1
            # Realizations of standard normal random variables.
            x_price = misc.normal_realizations(n_paths, rng, antithetic)
            # Increment of stock price process, and update.
            increment = \
                self._price_increment(paths[event_idx - 1], dt, x_price)
            paths[event_idx] = paths[event_idx - 1] + increment
        # Update.
        self.price_paths = paths


# TODO: Remove after tests!
class SDE:
    """SDE for the Black-Scholes model.

    Geometric Brownian motion is given by
        dS_t / S_t = (rate - dividend) * dt + vol * dW_t,
    which leads to
        S_t = S_0 exp((rate - dividend - vol^2 / 2) * t + vol * W_t).

    S_t is log-normally distributed
        ln(S_t) ~ N((rate - dividend - vol^2 / 2) * t, vol^2 * t).

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
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
