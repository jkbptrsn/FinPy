import math
import typing

import numpy as np
from scipy.stats import norm

from models import options
from models.black_scholes import misc
from utils import global_types
from utils import payoffs


class BinaryCashCall(options.Option1FAnalytical):
    """European cash-or-nothing call option in Black-Scholes model.

    European cash-or-nothing call option written on stock price modelled
    by Black-Scholes SDE. Pays out one unit of cash if the spot is above
    the strike at expiry.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            strike: float,
            expiry_idx: int,
            event_grid: np.ndarray,
            dividend: float = 0):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.model = global_types.Model.BLACK_SCHOLES
        self.type = global_types.Instrument.BINARY_CASH_CALL

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
        return payoffs.binary_cash_call(spot, self.strike)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        d2_deriv1 = 1 / (s * self.vol * math.sqrt(self.expiry - time))
        return math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2) \
            * d2_deriv1

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        d2_deriv1 = 1 / (s * self.vol * math.sqrt(self.expiry - time))
        d2_deriv2 = - 1 / (np.square(s) * self.vol
                           * math.sqrt(self.expiry - time))
        return math.exp(-self.rate * (self.expiry - time)) \
            * ((-d2) * norm.pdf(d2) * np.square(d2_deriv1)
               + norm.pdf(d2) * d2_deriv2)

    def rho(self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt rate.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        time = self.event_grid[event_idx]
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        d2_deriv1 = math.sqrt(self.expiry - time) / self.vol
        return (- (self.expiry - time)
                * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)
                + math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2)
                * d2_deriv1)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        d2_deriv1 = (d1 / (2 * (self.expiry - time))
                     - (self.rate + self.vol ** 2 / 2) /
                     (self.vol * math.sqrt(self.expiry - time))
                     + self.vol / (2 * math.sqrt(self.expiry - time)))
        return self.rate * self.price(spot, event_idx) \
            + math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2) \
            * d2_deriv1

    def vega(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt volatility.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        time = self.event_grid[event_idx]
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        d2_deriv1 = - d1 / self.vol
        return math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2) \
            * d2_deriv1

    def fd_solve(self) -> None:
        """Run finite difference solver on event_grid."""
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            self.fd.propagation(dt)


class BinaryAssetCall(options.Option1FAnalytical):
    """European asset-or-nothing call option in Black-Scholes model.

    European asset-or-nothing call option written on stock price
    modelled by Black-Scholes SDE. Pays out one unit of the asset if the
    spot is above the strike at expiry.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            strike: float,
            expiry_idx: int,
            event_grid: np.ndarray,
            dividend: float = 0):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.model = global_types.Model.BLACK_SCHOLES
        self.type = global_types.Instrument.BINARY_ASSET_CALL

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
        return payoffs.binary_asset_call(spot, self.strike)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return spot * norm.cdf(d1)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return s * norm.pdf(d1) \
            / (s * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(d1)

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
        pass

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
        pass

    def fd_solve(self) -> None:
        """Run finite difference solver on event_grid."""
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            self.fd.propagation(dt)


class BinaryCashPut(options.Option1FAnalytical):
    """European cash-or-nothing put option in Black-Scholes model.

    European cash-or-nothing put option written on stock price modelled
    by Black-Scholes SDE. Pays out one unit of cash if the spot is below
    the strike at expiry.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
    """

    def __init__(
            self,
            rate: float,
            vol: float,
            strike: float,
            expiry_idx: int,
            event_grid: np.ndarray,
            dividend: float = 0):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.model = global_types.Model.BLACK_SCHOLES
        self.type = global_types.Instrument.BINARY_CASH_PUT

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
        return payoffs.binary_cash_put(spot, self.strike)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return - math.exp(-self.rate * (self.expiry - time)) * norm.pdf(-d2) \
            / (s * self.vol * math.sqrt(self.expiry - time))

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
        pass

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
        pass

    def fd_solve(self) -> None:
        """Run finite difference solver on event_grid."""
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            self.fd.propagation(dt)


class BinaryAssetPut(options.Option1FAnalytical):
    """European asset-or-nothing put option in Black-Scholes model.

    European asset-or-nothing put option written on stock price modelled
    by Black-Scholes SDE. Pays out one unit of the asset if the spot is
    below the strike at expiry.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates as year fractions from as-of date.
        dividend: Continuous dividend yield. Default is 0.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 strike: float,
                 expiry_idx: int,
                 event_grid: np.ndarray,
                 dividend: float = 0):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.model = global_types.Model.BLACK_SCHOLES
        self.type = global_types.Instrument.BINARY_ASSET_PUT

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
        return payoffs.binary_asset_put(spot, self.strike)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return s * norm.cdf(-d1)

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
        s = spot * math.exp(-self.dividend * (self.expiry - time))
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return - s * norm.pdf(-d1) \
            / (s * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(-d1)

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
        pass

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
        pass

    def fd_solve(self) -> None:
        """Run finite difference solver on event_grid."""
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            self.fd.propagation(dt)
