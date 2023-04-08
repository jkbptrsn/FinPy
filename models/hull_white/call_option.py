import math
import typing

import numpy as np
from scipy.stats import norm

from models import options
from models.hull_white import misc as misc_hw
from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from models.hull_white import zero_coupon_bond as zcbond
from utils import global_types
from utils import misc
from utils import payoffs


class Call(sde.SDE, options.VanillaOption):
    """European call option class for the 1-factor Hull-White model.

    The European call option is written on a zero-coupon bond.
    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike value of underlying zero-coupon bond.
        expiry_idx: Event-grid index corresponding to expiry.
        maturity_idx: Event-grid index corresponding to maturity.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx

        self.option_type = global_types.Instrument.EUROPEAN_CALL

        # Underlying zero-coupon bond
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    maturity_idx, int_step_size=int_step_size)

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See proposition 4.5.1, L.B.G. Andersen & V.V. Piterbarg 2010.

        Args:
            spot: Spot value of PSEUDO short rate.
            event_idx: Event-grid index of current time.

        Returns:
            Call option price.
        """
        # Price of zero-coupon bond maturing at expiry_idx.
        self.zcbond.maturity_idx = self.expiry_idx
        bond_price_expiry = self.zcbond.price(spot, event_idx)
        # Price of zero-coupon bond maturing at maturity_idx.
        self.zcbond.maturity_idx = self.maturity_idx
        bond_price_maturity = self.zcbond.price(spot, event_idx)

        # Event-grid index corresponding to current time.
        int_event_idx1 = self.int_event_idx[event_idx]
        # Event-grid index corresponding to expiry.
        int_event_idx2 = self.int_event_idx[self.expiry_idx] + 1
        # Slice of integration grid.
        int_grid = self.int_grid[int_event_idx1:int_event_idx2]
        # Volatility strip on slice of integration grid.
        vol = self.vol_int_grid[int_event_idx1:int_event_idx2]
        # Constant kappa value.
        kappa = self.kappa.values[0]
        # v-function.
        integrand = vol ** 2 * np.exp(2 * kappa * int_grid)
        exp_kappa1 = math.exp(-kappa * self.event_grid[self.expiry_idx])
        exp_kappa2 = math.exp(-kappa * self.event_grid[self.maturity_idx])
        v = (exp_kappa1 - exp_kappa2) ** 2 \
            * np.sum(misc.trapz(int_grid, integrand)) / kappa ** 2
        # d-function.
        d = math.log(bond_price_maturity / (self.strike * bond_price_expiry))
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)
        return bond_price_maturity * norm.cdf(d_plus) \
            - self.strike * bond_price_expiry * norm.cdf(d_minus)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass


class CallNew(options.EuropeanOptionAnalytical1F):
    """European call option in 1-factor Hull-White model.

    European call option written on zero-coupon bond. See
    L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

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
            zcbond.ZCBondNew(kappa, vol, discount_curve, maturity_idx,
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
            spot: Current value of underlying.

        Returns:
            Payoff.
        """
        return payoffs.call(spot, self.strike)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current value of underlying.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        # P(t,T): Zero-coupon bond price at time zero with maturity T.
        self.zcbond.maturity_idx = self.expiry_idx
        self.zcbond.initialization()
        price1 = self.zcbond.price(spot, event_idx)
        # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
        self.zcbond.maturity_idx = self.maturity_idx
        self.zcbond.initialization()
        price2 = self.zcbond.price(spot, event_idx)
        # v-function.
        v = self.v_eg[event_idx]
        # d-function.
        d = np.log(price2 / (self.strike * price1))
#        d_plus = (d + v / 2) / np.sqrt(v)
#        d_minus = (d - v / 2) / np.sqrt(v)
#        return price2 * norm.cdf(d_plus) \
#            - self.strike * price1 * norm.cdf(d_minus)
        return np.zeros(spot.size)

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt value of underlying.

        Args:
            spot: Current value of underlying.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        pass

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt value of underlying.

        Args:
            spot: Current value of underlying.
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
            spot: Current value of underlying.
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

            # Propagation for one time step.
            self.fd.propagation(dt, True)

###############################################################################

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
