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


class Caplet(sde.SDE):
    """Caplet for the 1-factor Hull-White model.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Payment dates represented as year fractions from the
            as-of date.
        expiry_idx: Event grid index corresponding to expiry.
        maturity_idx: Event grid index corresponding to maturity.
        fixed_rate: Fixed rate.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 expiry_idx: int,
                 maturity_idx: int,
                 fixed_rate: float,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.fixed_rate = fixed_rate

        self.instrument_type = global_types.Instrument.CAPLET

        # Zero-coupon bond.
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    maturity_idx, int_step_size=int_step_size)

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See proposition 4.5.2, L.B.G. Andersen & V.V. Piterbarg 2010.

        Args:
            spot: Spot value of PSEUDO short rate.
            event_idx: Event-grid index of current time.

        Returns:
            Caplet price.
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
        tau = self.maturity - self.expiry
        d = math.log((1 + self.fixed_rate * tau)
                     * bond_price_maturity / bond_price_expiry)
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)
        return bond_price_expiry * norm.cdf(-d_minus) \
            - ((1 + self.fixed_rate * tau) * bond_price_maturity
               * norm.cdf(-d_plus))


class CapletNew(options.EuropeanOptionAnalytical1F):
    """Caplet in 1-factor Hull-White model.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        cap_rate: Capped value of simple forward rate.
        fixing_idx: Fixing index on event grid. Expiry of call option.
        payment_idx: Payment index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 cap_rate: float,
                 fixing_idx: int,
                 payment_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.cap_rate = cap_rate
        self.fixing_idx = fixing_idx
        self.payment_idx = payment_idx
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
        # y-function on event grid.
        self.y_eg = None
        # v-function on event grid.
        self.v_eg = None

        # Zero-coupon bond object used in analytical pricing.
        self.zcbond = \
            zcbond.ZCBondNew(kappa, vol, discount_curve, fixing_idx,
                             event_grid, time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.CAPLET

    # TODO: Expiry corresponds actually to the payment date.
    #  Maybe a new base call for options?
    @property
    def expiry(self) -> float:
        return self.event_grid[self.fixing_idx]

    @property
    def fixing_event(self) -> float:
        return self.event_grid[self.fixing_idx]

    @property
    def payment_event(self) -> float:
        return self.event_grid[self.payment_idx]

    @property
    def tenor(self) -> float:
        return self.payment_event - self.fixing_event

    def initialization(self):
        """Initialization of instrument object."""
        self.kappa_eg = self.zcbond.kappa_eg
        self.vol_eg = self.zcbond.vol_eg
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        self.y_eg = self.zcbond.y_eg
        # Kappa and vol are constant.
        if self.time_dependence == "constant":
            # v-function on event grid.
            self.v_eg = misc_hw.v_constant(self.zcbond.kappa_eg[0],
                                           self.zcbond.vol_eg[0],
                                           self.fixing_idx,
                                           self.payment_idx,
                                           self.event_grid)
        # Kappa is constant and vol is piecewise constant.
        elif self.time_dependence == "piecewise":
            # v-function on event grid.
            self.v_eg = misc_hw.v_piecewise(self.zcbond.kappa_eg[0],
                                            self.zcbond.vol_eg,
                                            self.fixing_idx,
                                            self.payment_idx,
                                            self.event_grid)
        else:
            raise ValueError(f"Time dependence unknown: "
                             f"{self.time_dependence}")

    def payoff(self,
               spot: typing.Union[float, np.ndarray],
               discounting: bool = False) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current value of pseudo short rate.
            discounting: ...

        Returns:
            Payoff at payment date.
        """
        # P(t_fixing, t_fixing). TODO: is one?
        price1 = self.zcbond_price(spot, self.fixing_idx, self.fixing_idx)
        # P(t_fixing, t_payment).
        price2 = self.zcbond_price(spot, self.fixing_idx, self.payment_idx)
        # Simple forward rate at t_fixing for (t_fixing, t_payment).
        simple_rate = self.simple_forward_rate(price2, self.tenor, price1)
        # Payoff.
        _payoff = self.tenor * np.maximum(simple_rate - self.cap_rate, 0)
        if discounting:
            return _payoff * price2 / price1
        else:
            return _payoff

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
        # P(t,T): Zero-coupon bond price at time t with maturity T.
        self.zcbond.maturity_idx = self.fixing_idx
        self.zcbond.initialization()
        price1 = self.zcbond.price(spot, event_idx)
        # P(t,T + tau): Zero-coupon bond price at time t with maturity
        # T + tau.
        self.zcbond.maturity_idx = self.payment_idx
        self.zcbond.initialization()
        price2 = self.zcbond.price(spot, event_idx)
        # Tenor.
        tenor = self.payment_event - self.fixing_event
        # v-function.
        v = self.v_eg[event_idx]
        # d-function.
        d = np.log((1 + self.cap_rate * tenor) * price2 / price1)
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)
        return price1 * norm.cdf(-d_minus) \
            - (1 + self.cap_rate * tenor) * price2 * norm.cdf(-d_plus)

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
        pass

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

        self.fd.solution = self.payoff(self.fd.grid, True)

        time_steps = np.flip(np.diff(self.event_grid[:self.fixing_idx + 1]))
        for count, dt in enumerate(time_steps):
            # Update drift, diffusion, and rate functions.
            event_idx = (self.fixing_idx - 1) - count
            drift = self.y_eg[event_idx] \
                - self.kappa_eg[event_idx] * self.fd.grid
            diffusion = self.vol_eg[event_idx] + 0 * self.fd.grid
            rate = self.fd.grid + self.forward_rate_eg[event_idx]
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)
            # Propagation for one time step dt.
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

    def zcbond_price(self,
                     spot: (float, np.ndarray),
                     event_idx: int,
                     maturity_idx: int) -> (float, np.ndarray):
        """Price of zero-coupon bond.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond price.
        """
        if self.zcbond.maturity_idx != maturity_idx:
            self.zcbond.maturity_idx = maturity_idx
            self.zcbond.initialization()
        return self.zcbond.price(spot, event_idx)

    @staticmethod
    def simple_forward_rate(bond_price_t2: (float, np.ndarray),
                            tau: float,
                            bond_price_t1: (float, np.ndarray) = 1.0) \
            -> (float, np.ndarray):
        """Calculate simple forward rate.

        The simple forward rate, from time t1 to time t2, at time t is
        defined as:
            (1 + (t2 - t1) * forward_rate(t, t1, t2)) =
                bond_price_t1(t) / bond_price_t2(t).
        See L.B.G. Andersen & V.V. Piterbarg 2010, section 4.1.

        Args:
            bond_price_t2: Zero-coupon bond price at time t2.
            tau: Time interval between t1 and t2.
            bond_price_t1: Zero-coupon bond price at time t1.
                Default is 1, in case that t > t1.

        Returns:
            Simple forward rate.
        """
        return (bond_price_t1 / bond_price_t2 - 1) / tau
