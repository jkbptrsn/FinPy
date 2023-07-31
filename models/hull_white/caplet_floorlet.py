import math
import typing

import numpy as np
from scipy.stats import norm

from models import options
from models.hull_white import misc as misc_hw
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types
from utils import misc


class CapletFloorlet(options.Option1FAnalytical):
    """Caplet or floorlet in 1-factor Hull-White model.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike_rate: Cap or floor rate.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        cap_or_floor: Caplet or floorlet. Default is caplet.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike_rate: float,
                 fixing_idx: int,
                 payment_idx: int,
                 event_grid: np.ndarray,
                 cap_or_floor: str = "caplet",
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike_rate = strike_rate
        self.fixing_idx = fixing_idx
        self.payment_idx = payment_idx
        self.event_grid = event_grid
        self.cap_or_floor = cap_or_floor
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
            zcbond.ZCBond(kappa, vol, discount_curve, fixing_idx,
                          event_grid, time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.transformation = global_types.Transformation.ANDERSEN
        if self.cap_or_floor == "caplet":
            self.type = global_types.Instrument.CAPLET
        elif self.cap_or_floor == "floorlet":
            self.type = global_types.Instrument.FLOORLET
        else:
            raise ValueError(f"Unknown instrument type: {self.cap_or_floor}")

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
                                           self.zcbond.g_eg,
                                           self.event_grid)
        # Kappa is constant and vol is piecewise constant.
        elif self.time_dependence == "piecewise":
            # v-function on event grid.
            self.v_eg = misc_hw.v_piecewise(self.zcbond.kappa_eg[0],
                                            self.zcbond.vol_eg,
                                            self.fixing_idx,
                                            self.payment_idx,
                                            self.zcbond.g_eg,
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
            discounting: Do analytical discounting from payment date to
                fixing date. Default is false.

        Returns:
            Payoff.
        """
        # P(t_fixing, t_payment).
        price = self.zcbond_price(spot, self.fixing_idx, self.payment_idx)
        # Simple forward rate at t_fixing for (t_fixing, t_payment).
        simple_rate = self.simple_forward_rate(price, self.tenor)
        # Payoff.
        if self.cap_or_floor == "caplet":
            _payoff = \
                self.tenor * np.maximum(simple_rate - self.strike_rate, 0)
        else:
            _payoff = \
                self.tenor * np.maximum(self.strike_rate - simple_rate, 0)
        # Do analytical discounting from payment date to fixing date.
        if discounting:
            return _payoff * price
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
        # P(t, t_fixing).
        price1 = self.zcbond_price(spot, event_idx, self.fixing_idx)
        # P(t, t_payment).
        price2 = self.zcbond_price(spot, event_idx, self.payment_idx)
        # v-function.
        v = self.v_eg[event_idx]
        # d-function.
        d = np.log((1 + self.strike_rate * self.tenor) * price2 / price1)
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)
        if self.cap_or_floor == "caplet":
            sign = 1
        else:
            sign = -1
        factor = (1 + self.strike_rate * self.tenor)
        return sign * price1 * norm.cdf(-sign * d_minus) \
            - sign * factor * price2 * norm.cdf(-sign * d_plus)

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
        # P(t, t_fixing).
        price1 = self.zcbond_price(spot, event_idx, self.fixing_idx)
        delta1 = self.zcbond_delta(spot, event_idx, self.fixing_idx)
        # P(t, t_payment).
        price2 = self.zcbond_price(spot, event_idx, self.payment_idx)
        delta2 = self.zcbond_delta(spot, event_idx, self.payment_idx)
        # v-function.
        v = self.v_eg[event_idx]
        # d-function.
        d = np.log((1 + self.strike_rate * self.tenor) * price2 / price1)
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)
        # Derivative of d-function.
        d_delta = (delta2 / price2 - delta1 / price1) / math.sqrt(v)
        if self.cap_or_floor == "caplet":
            sign = 1
        else:
            sign = -1
        factor = (1 + self.strike_rate * self.tenor)
        first_terms = sign * delta1 * norm.cdf(-sign * d_minus) \
            - sign * factor * delta2 * norm.cdf(-sign * d_plus)
        last_terms = sign * price1 * norm.pdf(-sign * d_minus) \
            - sign * factor * price2 * norm.pdf(-sign * d_plus)
        last_terms *= sign * d_delta
        return first_terms + last_terms

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
        # Payoff at payment event, discount to fixing event.
        self.fd.solution = self.payoff(self.fd.grid, True)
        # Numerical propagation from fixing event.
        time_steps = np.flip(np.diff(self.event_grid[:self.fixing_idx + 1]))
        for count, dt in enumerate(time_steps):
            # Event index before propagation with time step -dt.
            event_idx = self.fixing_idx - count
            # Update drift, diffusion, and rate functions.
            update_idx = event_idx - 1
            drift = self.y_eg[update_idx] \
                - self.kappa_eg[update_idx] * self.fd.grid
            diffusion = self.vol_eg[update_idx] + 0 * self.fd.grid
            rate = self.fd.grid + self.forward_rate_eg[update_idx]
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
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
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

    def zcbond_delta(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Delta of zero-coupon bond.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond delta.
        """
        if self.zcbond.maturity_idx != maturity_idx:
            self.zcbond.maturity_idx = maturity_idx
            self.zcbond.initialization()
        return self.zcbond.delta(spot, event_idx)

    @staticmethod
    def simple_forward_rate(bond_price_t2: typing.Union[float, np.ndarray],
                            tau: float,
                            bond_price_t1:
                            typing.Union[float, np.ndarray] = 1.0) \
            -> typing.Union[float, np.ndarray]:
        """Calculate simple forward rate.

        The simple forward rate at time t in (t1, t2) is defined as:
            (1 + (t2 - t1) * forward_rate(t, t1, t2)) =
                bond_price_t1(t) / bond_price_t2(t).
        See L.B.G. Andersen & V.V. Piterbarg 2010, section 4.1.

        Args:
            bond_price_t2: Price of zero-coupon bond with maturity t2.
            tau: Time interval between t1 and t2.
            bond_price_t1: Price of zero-coupon bond with maturity t1.
                Default is 1, in case that t > t1.

        Returns:
            Simple forward rate.
        """
        return (bond_price_t1 / bond_price_t2 - 1) / tau


class CapletFloorletPelsser(CapletFloorlet):
    """Caplet or floorlet in 1-factor Hull-White model.

    See Pelsser, chapter 5.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike_rate: Cap or floor rate.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        cap_or_floor: Caplet or floorlet. Default is caplet.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike_rate: float,
                 fixing_idx: int,
                 payment_idx: int,
                 event_grid: np.ndarray,
                 cap_or_floor: str = "caplet",
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         strike_rate,
                         fixing_idx,
                         payment_idx,
                         event_grid,
                         cap_or_floor,
                         time_dependence,
                         int_step_size)

        # Integration grid.
        self.int_grid = None
        # Indices of event dates on integration grid.
        self.int_event_idx = None
        # Speed of mean reversion on integration grid.
        self.kappa_ig = None
        # Step-wise integration of kappa on integration grid.
        self.int_kappa_step = None
        # Volatility on integration grid.
        self.vol_ig = None

        self.transformation = global_types.Transformation.PELSSER

        self.adjustment_rate = None
        self.adjustment_discount = None
        self.adjustment_function()

    def adjustment_function(self):
        """Adjustment of short rate transformation."""
        # P(0, t_{i+1}) / P(0, t_i)
        discount_steps = \
            self.discount_curve_eg[1:] / self.discount_curve_eg[:-1]
        discount_steps = np.append(1, discount_steps)
        # alpha_t_i - f(0,t_i), see Pelsser Eq (5.30).
        if self.time_dependence == "constant":
            alpha = misc_hw.alpha_constant(self.kappa_eg[0],
                                           self.vol_eg[0],
                                           self.event_grid)
            int_alpha = \
                misc_hw.int_alpha_constant(self.kappa_eg[0],
                                           self.vol_eg[0],
                                           self.event_grid)
        elif self.time_dependence == "piecewise":
            alpha = misc_hw.alpha_piecewise(self.kappa_eg[0],
                                            self.vol_eg,
                                            self.event_grid)
            int_alpha = \
                misc_hw.int_alpha_piecewise(self.kappa_eg[0],
                                            self.vol_eg,
                                            self.event_grid)
        elif self.time_dependence == "general":
            self.int_grid, self.int_event_idx = \
                misc_hw.integration_grid(self.event_grid, self.int_step_size)
            # Speed of mean reversion interpolated on integration grid.
            self.kappa_ig = self.kappa.interpolation(self.int_grid)
            # Volatility interpolated on integration grid.
            self.vol_ig = self.vol.interpolation(self.int_grid)
            # Integration of speed of mean reversion using trapezoidal rule.
            self.int_kappa_step = \
                np.append(0, misc.trapz(self.int_grid, self.kappa_ig))
            alpha = \
                misc_hw.alpha_general(self.int_grid,
                                      self.int_event_idx,
                                      self.int_kappa_step,
                                      self.vol_ig,
                                      self.event_grid)
            int_alpha = \
                misc_hw.int_alpha_general(self.int_grid,
                                          self.int_event_idx,
                                          self.int_kappa_step,
                                          self.vol_ig,
                                          self.event_grid)
        else:
            raise ValueError(f"Time-dependence is unknown: "
                             f"{self.time_dependence}")
        # TODO: Adjustment from Pelsser to Andersen.
        self.adjustment_rate = alpha
        # Adjustment to real discount factor.
        self.adjustment_discount = discount_steps * np.exp(-int_alpha)

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()

        # TODO: Transformation adjustment is incorrect!

        # Payoff at payment event.
        grid = self.fd.grid + self.adjustment_rate[self.fixing_idx]
        self.fd.solution = self.payoff(grid, False)

        # Numerical propagation from payment event.
        time_steps = np.flip(np.diff(self.event_grid[:self.payment_idx + 1]))
        for count, dt in enumerate(time_steps):
            # Event index before propagation with time step -dt.
            event_idx = self.payment_idx - count
            # Update drift, diffusion, and rate functions.
            update_idx = event_idx - 1
            drift = -self.kappa_eg[update_idx] * self.fd.grid
            diffusion = self.vol_eg[update_idx] + 0 * self.fd.grid
            rate = self.fd.grid
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)
            # Propagation for one time step dt.
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjustment_discount[event_idx]
