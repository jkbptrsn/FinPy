import abc
import math
import typing

import numpy as np

from utils import global_types
from utils import misc


class Sde(metaclass=abc.ABCMeta):
    """SDE for short rate process in Vasicek model.

    The short rate r_t is defined by
        dr_t = kappa * (mean_rate - r_t) * dt + vol * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    See Andersen & Piterbarg (2010), Section 10.1.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
    """

    def __init__(
            self,
            kappa: float,
            mean_rate: float,
            vol: float,
            event_grid: np.ndarray):
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.event_grid = event_grid

        self.model = global_types.Model.VASICEK

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
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        pass


class SdeExact(Sde):
    """SDE for short rate process in Vasicek model.

    The short rate r_t is defined by
        dr_t = kappa * (mean_rate - r_t) * dt + vol * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    See Andersen & Piterbarg (2010), Chapter 10.1.

    Monte-Carlo paths constructed using exact discretization.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
    """

    def __init__(
            self,
            kappa: float,
            mean_rate: float,
            vol: float,
            event_grid: np.ndarray):
        super().__init__(kappa, mean_rate, vol, event_grid)

        # Arrays used for discretization.
        self.rate_mean = np.zeros((self.event_grid.size, 2))
        self.rate_variance = np.zeros(self.event_grid.size)
        self.discount_mean = np.zeros((self.event_grid.size, 2))
        self.discount_variance = np.zeros(self.event_grid.size)
        self.covariance = np.zeros(self.event_grid.size)

        self.rate_paths = None
        self.discount_paths = None
        self.mc_estimate = None
        self.mc_error = None

        self.initialization()

    def initialization(self) -> None:
        """Initialization of Monte-Carlo solver."""
        self._calc_rate_mean()
        self._calc_rate_variance()
        self._calc_discount_mean()
        self._calc_discount_variance()
        self._calc_covariance()

    def _calc_rate_mean(self) -> None:
        """Conditional mean of short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.12).
        """
        exp_kappa = np.exp(-self.kappa * np.diff(self.event_grid))
        self.rate_mean[0, 0] = 1
        self.rate_mean[1:, 0] = exp_kappa
        self.rate_mean[1:, 1] = self.mean_rate * (1 - exp_kappa)

    def _calc_rate_variance(self) -> None:
        """Conditional variance of short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.13).
        """
        two_kappa = 2 * self.kappa
        exp_two_kappa = np.exp(-two_kappa * np.diff(self.event_grid))
        self.rate_variance[1:] = \
            self.vol ** 2 * (1 - exp_two_kappa) / two_kappa

    def _rate_increment(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment short rate process one time step.

        The spot rate is subtracted to get the increment.

        Args:
            spot: Short rate at event corresponding to event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Increment of short rate process.
        """
        mean = \
            self.rate_mean[event_idx, 0] * spot + self.rate_mean[event_idx, 1]
        variance = self.rate_variance[event_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def _calc_discount_mean(self) -> None:
        """Conditional mean of discount process.

        Here the discount process refers to -int_{t_1}^{t_2} r_t dt.

        See Andersen & Piterbarg (2010), Eq. (10.12+).
        """
        dt = np.diff(self.event_grid)
        exp_kappa = np.exp(-self.kappa * dt)
        exp_kappa = (1 - exp_kappa) / self.kappa
        self.discount_mean[1:, 0] = -exp_kappa
        self.discount_mean[1:, 1] = self.mean_rate * (exp_kappa - dt)

    def _calc_discount_variance(self) -> None:
        """Conditional variance of discount process.

        Here the discount process refers to -int_{t_1}^{t_2} r_t dt.

        See Andersen & Piterbarg (2010), Eq. (10.13+).
        """
        dt = np.diff(self.event_grid)
        exp_kappa = np.exp(-self.kappa * dt)
        two_kappa = 2 * self.kappa
        exp_two_kappa = np.exp(-two_kappa * dt)
        self.discount_variance[1:] = \
            self.vol ** 2 * (4 * exp_kappa - 3 + two_kappa * dt
                             - exp_two_kappa) / (2 * self.kappa ** 3)

    def _discount_increment(
            self,
            spot_rate: typing.Union[float, np.ndarray],
            event_idx: int,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment discount process one time step.

        Args:
            spot_rate: Short rate at event corresponding to
                event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Increment of discount process.
        """
        mean = self.discount_mean[event_idx, 0] * spot_rate \
            + self.discount_mean[event_idx, 1]
        variance = self.discount_variance[event_idx]
        return mean + math.sqrt(variance) * normal_rand

    def _calc_covariance(self) -> None:
        """Conditional covariance of short rate and discount processes.

        See Andersen & Piterbarg (2010), Lemma 10.1.11.
        """
        dt = np.diff(self.event_grid)
        vol_sq = self.vol ** 2
        kappa_sq = self.kappa ** 2
        exp_kappa = np.exp(-self.kappa * dt)
        exp_two_kappa = np.exp(-2 * self.kappa * dt)
        self.covariance[1:] = \
            vol_sq * (2 * exp_kappa - exp_two_kappa - 1) / (2 * kappa_sq)

    def _correlation(
            self,
            event_idx: int) -> float:
        """Conditional correlation of short rate and discount processes.

        Args:
            event_idx: Index on event grid.

        Returns:
            Conditional correlation.
        """
        covariance = self.covariance[event_idx]
        rate_var = self.rate_variance[event_idx]
        discount_var = self.discount_variance[event_idx]
        return covariance / math.sqrt(rate_var * discount_var)

    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False):
        """Generation of Monte-Carlo paths using exact discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        # Paths of rate process.
        r_paths = np.zeros((self.event_grid.size, n_paths))
        r_paths[0] = spot
        # Paths of discount process.
        d_paths = np.zeros((self.event_grid.size, n_paths))
        for event_idx in range(1, self.event_grid.size):
            correlation = self._correlation(event_idx)
            # Realizations of standard normal random variables.
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, rng, antithetic)
            # Increment of rate process, and update.
            r_increment = self._rate_increment(r_paths[event_idx - 1],
                                               event_idx, x_rate)
            r_paths[event_idx] = r_paths[event_idx - 1] + r_increment
            # Increment of discount process, and update.
            d_increment = self._discount_increment(r_paths[event_idx - 1],
                                                   event_idx, x_discount)
            d_paths[event_idx] = d_paths[event_idx - 1] + d_increment
        # Get actual discount factors on event_grid.
        d_paths = np.exp(d_paths)
        # Update.
        self.rate_paths = r_paths
        self.discount_paths = d_paths


class SdeEuler(Sde):
    """SDE for short rate process in Vasicek model.

    The short rate r_t is defined by
        dr_t = kappa * (mean_rate - r_t) * dt + vol * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    See Andersen & Piterbarg (2010), Chapter 10.1.

    Monte-Carlo paths constructed using Euler-Maruyama discretization.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.
    """

    def __init__(
            self,
            kappa: float,
            mean_rate: float,
            vol: float,
            event_grid: np.ndarray):
        super().__init__(kappa, mean_rate, vol, event_grid)

        self.rate_paths = None
        self.discount_paths = None
        self.mc_estimate = None
        self.mc_error = None

    def _rate_increment(
            self,
            spot: typing.Union[float, np.ndarray],
            dt: float,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment short rate process one time step.

        Args:
            spot: Short rate at event corresponding to event_idx - 1.
            dt: Time step.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Increment of short rate process.
        """
        wiener_increment = math.sqrt(dt) * normal_rand
        rate_increment = self.kappa * (self.mean_rate - spot) * dt \
            + self.vol * wiener_increment
        return rate_increment

    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False):
        """Generation of Monte-Carlo paths using Euler discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        # Paths of rate process.
        r_paths = np.zeros((self.event_grid.size, n_paths))
        r_paths[0] = spot
        # Paths of discount process.
        d_paths = np.zeros((self.event_grid.size, n_paths))
        for idx, dt in enumerate(np.diff(self.event_grid)):
            event_idx = idx + 1
            # Realizations of standard normal random variables.
            x_rate = misc.normal_realizations(n_paths, rng, antithetic)
            # Increment of rate process, and update.
            r_increment = \
                self._rate_increment(r_paths[event_idx - 1], dt, x_rate)
            r_paths[event_idx] = r_paths[event_idx - 1] + r_increment
            # Increment of discount process, and update.
            d_increment = -r_paths[event_idx - 1] * dt
            d_paths[event_idx] = d_paths[event_idx - 1] + d_increment
        # Get actual discount factors on event_grid.
        d_paths = np.exp(d_paths)
        # Update.
        self.rate_paths = r_paths
        self.discount_paths = d_paths
