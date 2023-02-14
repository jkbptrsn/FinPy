import math
import numpy as np
from scipy.stats import norm

import models.options as options
import models.bachelier.misc as misc
import models.bachelier.sde as sde

from numerical_methods.finite_difference import theta as fd_theta

import utils.global_types as global_types
import utils.payoffs as payoffs


class CallNew(options.VanillaOptionNew):
    """European call option in Bachelier model.

    European call option written on stock price.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 strike: float,
                 expiry_idx: int,
                 event_grid: np.ndarray):
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid

        self.type = global_types.InstrumentType.EUROPEAN_CALL
        self.model = global_types.ModelName.BACHELIER
        self.fd = None
        self.mc = None

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.call(spot, self.strike)

    def payoff_dds(self,
                   spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        dn = misc.dn(spot, time, self.expiry, self.strike, self.vol)
        # Time-to-maturity
        ttm = self.expiry - time
        # Discount factor
        discount = math.exp(-self.rate * ttm)
        return discount \
            * ((spot - self.strike) * norm.cdf(dn)
               + self.vol * math.sqrt(self.expiry - time) * norm.pdf(dn))

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def rho(self,
            spot: (float, np.ndarray),
            time: float) -> (float, np.ndarray):
        """..."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        """..."""
        pass

    def fd_setup(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 theta_value: float = 0.5,
                 method: str = "Andersen"):
        """Setting up finite difference solver.
        TODO: Add non-equidistant grid. Instead of xmin, xmax, nstates, use state_grid as parameter
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


class Call(sde.SDE):
    """European call option in Bachelier model.

    European call option written on stock price.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.EUROPEAN_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.call(spot, self.strike)

    def payoff_dds(self,
                   spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        dn = misc.dn(spot, time, self.expiry, self.strike, self.vol)
        # Time-to-maturity
        ttm = self.expiry - time
        # Discount factor
        discount = math.exp(-self.rate * ttm)
        return discount \
            * ((spot - self.strike) * norm.cdf(dn)
               + self.vol * math.sqrt(self.expiry - time) * norm.pdf(dn))

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def rho(self,
            spot: (float, np.ndarray),
            time: float) -> (float, np.ndarray):
        """..."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        """..."""
        pass
