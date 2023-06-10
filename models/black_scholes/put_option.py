import math
import typing

import numpy as np
from scipy.stats import norm

from models import options
from models.black_scholes import misc
from models.black_scholes import sde
from utils import global_types
from utils import payoffs
from utils import smoothing


class Put(options.EuropeanOptionAnalytical1F):
    """European put option in Black-Scholes model.

    European put option written on stock price modelled by
    Black-Scholes SDE.

    See J.C. Hull 2015, chapter 15 and 19.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        dividend: Continuous dividend yield. Default value is 0.
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
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.type = global_types.Instrument.EUROPEAN_PUT
        self.model = global_types.Model.BLACK_SCHOLES

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        return payoffs.put(spot, self.strike)

    def payoff_dds(self,
                   spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Derivative of payoff function wrt value of underlying.

        1st order partial derivative of payoff function wrt value of
        underlying.

        Args:
            spot: Current stock price.

        Returns:
            Derivative of payoff.
        """
        return - payoffs.binary_cash_put(spot, self.strike)

    def price(self,
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
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return - s * norm.cdf(-d1) \
            + self.strike * norm.cdf(-d2) * math.exp(-self.rate * delta_t)

    def delta(self,
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
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return math.exp(-self.dividend * delta_t) * (norm.cdf(d1) - 1)

    def gamma(self,
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
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
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
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return - self.strike * delta_t * math.exp(-self.rate * delta_t) \
            * norm.cdf(-d2)

    def theta(self,
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
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return - s * norm.pdf(d1) * self.vol / (2 * math.sqrt(delta_t)) \
            + self.rate * self.strike * math.exp(-self.rate * delta_t) \
            * norm.cdf(-d2) - self.dividend * s * norm.cdf(-d1)

    def vega(self,
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
        s = spot * math.exp(-self.dividend * delta_t)
        d1, d2 = \
            misc.d1d2(s, time, self.rate, self.vol, self.expiry, self.strike)
        return s * norm.pdf(d1) * math.sqrt(delta_t)

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.set_propagator()
            self.fd.propagation(dt)

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        self.mc_exact = \
            sde.SDE(self.rate, self.vol, self.event_grid, self.dividend)

    def mc_exact_solve(self):
        """Run Monte-Carlo solver on event_grid."""
        self.mc_exact.paths()


class PutAmerican(options.AmericanOption):
    """American put option in Black-Scholes model.

    TODO:
     * MC pricing of European put option.
     * MC pricing of American put option using LSM.
     * FD pricing of American call option.
     * MC pricing of European call option.
     * MC pricing of American put option using LSM.

    American put option written on stock price modelled by
    Black-Scholes SDE.

    See J.C. Hull 2015, chapter 15 and 19.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        dividend: Continuous dividend yield. Default value is 0.
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
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.type = global_types.Instrument.AMERICAN_PUT
        self.model = global_types.Model.BLACK_SCHOLES

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        return payoffs.put(spot, self.strike)

    def payoff_dds(self,
                   spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Derivative of payoff function wrt value of underlying.

        1st order partial derivative of payoff function wrt value of
        underlying.

        Args:
            spot: Current stock price.

        Returns:
            Derivative of payoff.
        """
        return - payoffs.binary_cash_put(spot, self.strike)

    def fd_solve(self):
        """Run solver on event_grid..."""
        counter = 0
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.set_propagator()
            self.fd.propagation(dt)

            # Compare continuation value and exercise value.
            if counter % 10 == 0:
                self.fd.solution = \
                    np.maximum(self.fd.solution, self.strike - self.fd.grid)
#                self.fd.solution = \
#                    smoothing.smoothing_1d(self.fd.grid, self.fd.solution)

            counter += 1
