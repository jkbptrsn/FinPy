import typing

import numpy as np

from models import options
from models.hull_white import misc as misc_hw
from models.hull_white import zero_coupon_bond as zcbond
from utils import global_types
from utils import misc
from utils import payoffs


class Call(options.EuropeanOptionAnalytical1F):
    """European call option in 1-factor Hull-White model.

    European call option written on zero-coupon bond. See
    L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike: Strike value of underlying zero-coupon bond.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_step_size = int_step_size

        # Speed of mean reversion on event grid.
        self.kappa_eg = None
        # Volatility on event grid.
        self.vol_eg = None
        # Discount curve on event grid.
        self.discount_curve_eg = None
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = None
        # G-function on event grid.
        self.g_eg = None
        # y-function on event grid.
        self.y_eg = None
        # v-function on event grid.
        self.v_eg = None
        # Underlying zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBond(kappa, vol, discount_curve, maturity_idx,
                          event_grid, time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.EUROPEAN_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.zcbond.maturity

    def initialization(self):
        """Initialization of instrument object."""

        self.kappa_eg = self.zcbond.kappa_eg
        self.vol_eg = self.zcbond.vol_eg
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        self.g_eg = self.zcbond.g_eg
        self.y_eg = self.zcbond.y_eg

        # Kappa and vol are constant.
        if self.time_dependence == "constant":
            # v-function on event grid.
            self.v_eg = misc_hw.v_constant(self.zcbond.kappa_eg[0],
                                           self.zcbond.vol_eg[0],
                                           self.expiry_idx,
                                           self.maturity_idx,
                                           self.event_grid)
        # Kappa is constant and vol is piecewise constant.
        elif self.time_dependence == "piecewise":
            # v-function on event grid.
            self.v_eg = misc_hw.v_piecewise(self.zcbond.kappa_eg[0],
                                            self.zcbond.vol_eg,
                                            self.expiry_idx,
                                            self.maturity_idx,
                                            self.event_grid)
        else:
            raise ValueError(f"Time dependence unknown: "
                             f"{self.time_dependence}")

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current value of underlying zero-coupon bond.

        Returns:
            Payoff.
        """
        return payoffs.call(spot, self.strike)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return misc_hw.call_put_price(spot, self.strike, event_idx,
                                      self.expiry_idx, self.maturity_idx,
                                      self.zcbond, self.v_eg, "call")

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt value of underlying.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        return misc_hw.call_put_delta(spot, self.strike, event_idx,
                                      self.expiry_idx, self.maturity_idx,
                                      self.zcbond, self.v_eg, "call")

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt value of underlying.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        pass

    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()

        self.fd.solution = self.zcbond.payoff(self.fd.grid)

        for count, dt in enumerate(np.flip(np.diff(self.event_grid))):
            # Time index at time "t", when moving from "t+1" to "t".
            idx = -2 - count
            # Update drift, diffusion, and rate functions.
            drift = self.y_eg[idx] - self.kappa_eg[idx] * self.fd.grid
            diffusion = self.vol_eg[idx] + 0 * self.fd.grid
            rate = self.fd.grid + self.forward_rate_eg[idx]
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)

            # Option payoff at expiry...
            if -(idx + 1) == (self.maturity_idx - self.expiry_idx + 1):
                self.fd.solution = self.payoff(self.fd.solution)

            # Propagation for one time step.
            self.fd.propagation(dt, True)

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        pass

    def mc_exact_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of short rate and discount processes
            represented on event grid.
        """
        pass
