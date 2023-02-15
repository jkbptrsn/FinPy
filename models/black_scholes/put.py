import math
import numpy as np
from scipy.stats import norm

import models.options as options
import models.black_scholes.misc as misc
import models.black_scholes.sde as sde

from numerical_methods.finite_difference import theta as fd_theta

import utils.global_types as global_types
import utils.payoffs as payoffs


class PutNew(options.VanillaOptionNew):
    """European put option in Black-Scholes model.

    European put option written on stock price.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        dividend: Stock dividend.

        # TODO: Add basic unit tests of class.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 strike: float,
                 expiry_idx: int,
                 event_grid: np.ndarray,
                 dividend: float = 0):
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid
        self.dividend = dividend

        self.type = global_types.Instrument.EUROPEAN_PUT
        self.model = global_types.Model.BLACK_SCHOLES
        self.fd = None
        self.mc = None

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        return payoffs.put(spot, self.strike)

    def payoff_dds(self,
                   spot: (float, np.ndarray)) -> (float, np.ndarray):
        """1st order partial derivative of payoff function wrt the
        underlying state.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        return - payoffs.binary_cash_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        time = self.event_grid[event_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= np.exp(-self.dividend * (self.expiry - time))
        return - spot * norm.cdf(-d1) \
            + self.strike * norm.cdf(-d2) \
            * math.exp(-self.rate * (self.expiry - time))

    def delta(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt stock price.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        time = self.event_grid[event_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return np.exp(-self.dividend * (self.expiry - time)) \
            * (norm.cdf(d1) - 1)

    def gamma(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """2nd order price sensitivity wrt stock price.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        time = self.event_grid[event_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return math.exp(-self.dividend * (self.expiry - time)) * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def rho(self,
            spot: (float, np.ndarray),
            event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt rate.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Rho.
        """
        time = self.event_grid[event_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return - self.strike * (self.expiry - time) \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def theta(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt time.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        time = self.event_grid[event_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= math.exp(-self.dividend * (self.expiry - time))
        return - spot * norm.pdf(d1) * self.vol \
            / (2 * math.sqrt(self.expiry - time)) \
            + self.rate * self.strike \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2) \
            - self.dividend * spot * norm.cdf(d1)

    def vega(self,
             spot: (float, np.ndarray),
             event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt volatility.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Vega.
        """
        time = self.event_grid[event_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return spot * norm.pdf(d1) * math.sqrt(self.expiry - time)

    def fd_setup(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 theta_value: float = 0.5,
                 method: str = "Andersen"):
        """Setting up finite difference solver.

        Args:
            xmin: Minimum of stock price range.
            xmax: Maximum of stock price range.
            nstates: Number of states.
            theta_value: ...
            method: "Andersen" og "Andreasen"

        Returns:
            Finite difference solver.
        """
        self.fd = fd_theta.setup_solver(xmin, xmax, nstates,
                                        self, theta_value, method)
        self.fd.initialization()

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            # TODO: Use dt in propagation, with non-equidistant event grid...
            # Will this work for both theta-method implementations?
            self.fd.set_propagator()
            self.fd.propagation()


class Put(sde.SDE, options.VanillaOption):
    """European put option in Black-Scholes model.

    European put option written on stock price.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        dividend: Stock dividend.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.Instrument.EUROPEAN_PUT

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               state: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            state: State of underlying process.

        Returns:
            Payoff.
        """
        return payoffs.put(state, self.strike)

    def payoff_dds(self,
                   state: (float, np.ndarray)) -> (float, np.ndarray):
        """1st order partial derivative of payoff function wrt the
        underlying state.

        Args:
            state: State of underlying process.

        Returns:
            Payoff.
        """
        return - payoffs.binary_cash_put(state, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """Price function.

        Args:
            spot: Spot price.
            time_idx: Index on event grid.

        Returns:
            Price.
        """
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= np.exp(-self.dividend * (self.expiry - time))
        return - spot * norm.cdf(-d1) \
            + self.strike * norm.cdf(-d2) \
            * math.exp(-self.rate * (self.expiry - time))

    def delta(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state.

        Args:
            spot: Spot price.
            time_idx: Index on event grid.

        Returns:
            Delta.
        """
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return np.exp(-self.dividend * (self.expiry - time)) \
            * (norm.cdf(d1) - 1)

    def gamma(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state.

        Args:
            spot: Spot price.
            time_idx: Index on event grid.

        Returns:
            Gamma.
        """
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return math.exp(-self.dividend * (self.expiry - time)) * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def rho(self,
            spot: (float, np.ndarray),
            time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt rate.

        Args:
            spot: Spot price.
            time_idx: Index on event grid.

        Returns:
            Rho.
        """
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return - self.strike * (self.expiry - time) \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def theta(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot price.
            time_idx: Index on event grid.

        Returns:
            Theta.
        """
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= math.exp(-self.dividend * (self.expiry - time))
        return - spot * norm.pdf(d1) * self.vol \
            / (2 * math.sqrt(self.expiry - time)) \
            + self.rate * self.strike \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2) \
            - self.dividend * spot * norm.cdf(d1)

    def vega(self,
             spot: (float, np.ndarray),
             time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt volatility.

        Args:
            spot: Spot price.
            time_idx: Index on event grid.

        Returns:
            Vega.
        """
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return spot * norm.pdf(d1) * math.sqrt(self.expiry - time)
