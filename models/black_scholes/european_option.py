import math
import typing

import numpy as np
from scipy.stats import norm

from models import options
from models.black_scholes import misc
from models.black_scholes import sde
from numerics.mc import lsm
from utils import global_types
from utils import payoffs


class EuropeanOption(options.Option1FAnalytical):
    """European call/put option in Black-Scholes model.

    European call/put option written on stock price modelled by
    Black-Scholes SDE.

    See Hull (2015), Chapters 15 and 19.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
        type_: Option type. Default is call.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            strike: float,
            expiry_idx: int,
            event_grid: np.ndarray,
            dividend: float = 0,
            type_: str = "Call"):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.model = global_types.Model.BLACK_SCHOLES
        if type_ == "Call":
            self.type = global_types.Instrument.EUROPEAN_CALL
        elif type_ == "Put":
            self.type = global_types.Instrument.EUROPEAN_PUT
        else:
            raise ValueError(f"Unknown option type: {type_}")

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return payoffs.call(spot, self.strike)
        else:
            return payoffs.put(spot, self.strike)

    def payoff_dds(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Derivative of payoff function wrt stock price.

        Args:
            spot: Current stock price.

        Returns:
            Derivative of payoff.
        """
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return payoffs.binary_cash_call(spot, self.strike)
        else:
            return -payoffs.binary_cash_put(spot, self.strike)

    def price(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        time = self.event_grid[event_idx]
        delta_t = self.expiry - time
        # Correct for dividends.
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return s * norm.cdf(d1) \
                - self.strike * norm.cdf(d2) * math.exp(-self.rate * delta_t)
        else:
            return - s * norm.cdf(-d1) \
                + self.strike * norm.cdf(-d2) * math.exp(-self.rate * delta_t)

    def delta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt stock price.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        time = self.event_grid[event_idx]
        delta_t = self.expiry - time
        # Correct for dividends.
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return math.exp(-self.dividend * delta_t) * norm.cdf(d1)
        else:
            return math.exp(-self.dividend * delta_t) * (norm.cdf(d1) - 1)

    def gamma(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt stock price.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        time = self.event_grid[event_idx]
        delta_t = self.expiry - time
        # Correct for dividends.
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return math.exp(-self.dividend * delta_t) * norm.pdf(d1) \
                / (spot * self.vol * math.sqrt(delta_t))
        else:
            return math.exp(-self.dividend * delta_t) * norm.pdf(d1) \
                / (spot * self.vol * math.sqrt(delta_t))

    def rho(self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt rate.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Rho.
        """
        time = self.event_grid[event_idx]
        delta_t = self.expiry - time
        # Correct for dividends.
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return self.strike * delta_t * math.exp(-self.rate * delta_t) \
                * norm.cdf(d2)
        else:
            return - self.strike * delta_t * math.exp(-self.rate * delta_t) \
                * norm.cdf(-d2)

    def theta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        time = self.event_grid[event_idx]
        delta_t = self.expiry - time
        # Correct for dividends.
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return - s * norm.pdf(d1) * self.vol / (2 * math.sqrt(delta_t)) \
                - self.rate * self.strike * math.exp(-self.rate * delta_t) \
                * norm.cdf(d2) + self.dividend * s * norm.cdf(d1)
        else:
            return - s * norm.pdf(d1) * self.vol / (2 * math.sqrt(delta_t)) \
                + self.rate * self.strike * math.exp(-self.rate * delta_t) \
                * norm.cdf(-d2) - self.dividend * s * norm.cdf(-d1)

    def vega(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt volatility.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Vega.
        """
        time = self.event_grid[event_idx]
        delta_t = self.expiry - time
        # Correct for dividends.
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        if self.type == global_types.Instrument.EUROPEAN_CALL:
            return s * norm.pdf(d1) * math.sqrt(delta_t)
        else:
            return s * norm.pdf(d1) * math.sqrt(delta_t)

    def fd_solve(self) -> None:
        """Run finite difference solver on event_grid."""
        # Backward propagation.
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.propagation(dt)

    def mc_exact_setup(self) -> None:
        """Setup exact Monte-Carlo solver."""
        self.mc_exact = \
            sde.SdeExact(self.rate, self.vol, self.event_grid, self.dividend)

    def mc_exact_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Exact discretization.

        Args:
            spot: Spot stock price.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        # Stock price at expiry.
        prices = self.mc_exact.price_paths[self.expiry_idx]
        # Option payoffs.
        option_prices = self.payoff(prices)
        # Discounted payoffs.
        option_prices *= self.mc_exact.discount_grid[self.expiry_idx]
        self.mc_exact.mc_estimate = option_prices.mean()
        self.mc_exact.mc_error = option_prices.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self) -> None:
        """Setup Euler Monte-Carlo solver."""
        self.mc_euler = \
            sde.SdeEuler(self.rate, self.vol, self.event_grid, self.dividend)

    def mc_euler_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Spot stock price.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        # Stock price at expiry.
        prices = self.mc_euler.price_paths[self.expiry_idx]
        # Option payoffs.
        option_prices = self.payoff(prices)
        # Discounted payoffs.
        option_prices *= self.mc_euler.discount_grid[self.expiry_idx]
        self.mc_euler.mc_estimate = option_prices.mean()
        self.mc_euler.mc_error = option_prices.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)


class AmericanOption(options.Option1F):
    """American call/put option in Black-Scholes model.

    American call/put option written on stock price modelled by
    Black-Scholes SDE.

    See Hull (2015), Chapters 15 and 19.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        exercise_grid: Exercise indices on event_grid.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
        type_: Option type. Default is call.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            strike: float,
            exercise_grid: np.array,
            event_grid: np.ndarray,
            dividend: float = 0,
            type_: str = "Call"):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.exercise_grid = exercise_grid
        self.event_grid = event_grid
        self.dividend = dividend

        self.model = global_types.Model.BLACK_SCHOLES
        if type_ == "Call":
            self.type = global_types.Instrument.AMERICAN_CALL
        elif type_ == "Put":
            self.type = global_types.Instrument.AMERICAN_PUT
        else:
            raise ValueError(f"Unknown option type: {type_}")

    @property
    def expiry(self) -> float:
        return self.event_grid[self.exercise_grid[-1]]

    def payoff(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        if self.type == global_types.Instrument.AMERICAN_CALL:
            return payoffs.call(spot, self.strike)
        else:
            return payoffs.put(spot, self.strike)

    def payoff_dds(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Derivative of payoff function wrt stock price.

        Args:
            spot: Current stock price.

        Returns:
            Derivative of payoff.
        """
        if self.type == global_types.Instrument.AMERICAN_CALL:
            return payoffs.binary_cash_call(spot, self.strike)
        else:
            return -payoffs.binary_cash_put(spot, self.strike)

    def fd_solve(self) -> None:
        """Run finite difference solver on event_grid."""
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - idx
            # Compare continuation value and exercise value.
            if event_idx in self.exercise_grid:
                exercise_value = self.payoff(self.fd.grid)
                self.fd.solution = np.maximum(self.fd.solution, exercise_value)
            self.fd.propagation(dt)

    def mc_exact_setup(self) -> None:
        """Setup exact Monte-Carlo solver."""
        self.mc_exact = \
            sde.SdeExact(self.rate, self.vol, self.event_grid, self.dividend)

    def mc_exact_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Exact discretization.

        Args:
            spot: Spot stock price.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        self.mc_exact.mc_estimate, self.mc_exact.mc_error = (
            lsm.black_scholes(self))

    def mc_euler_setup(self) -> None:
        """Setup Euler Monte-Carlo solver."""
        self.mc_euler = \
            sde.SdeEuler(self.rate, self.vol, self.event_grid, self.dividend)

    def mc_euler_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Spot stock price.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        self.mc_euler.mc_estimate, self.mc_euler.mc_error = (
            lsm.black_scholes(self))
