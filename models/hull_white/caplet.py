import math
import typing

import numpy as np

from models import options
from models.hull_white import misc_caplet as misc_cf
from models.hull_white import misc_european_option as misc_ep
from models.hull_white import misc_swap as misc_sw
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types
from utils import payoffs


class Caplet(options.Option1FAnalytical):
    """Caplet or floorlet in 1-factor Hull-White model.

    Price of caplet of floorlet.

    See Andersen & Piterbarg (2010), Proposition 4.5.2, and
    Brigo & Mercurio (2007), Section 3.3.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike_rate: Caplet or floorlet rate.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        option_type: Caplet or floorlet. Default is caplet.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike_rate: float,
                 fixing_idx: int,
                 payment_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 option_type: str = "caplet"):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike_rate = strike_rate
        self.fixing_idx = fixing_idx
        self.payment_idx = payment_idx
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBond(kappa,
                          vol,
                          discount_curve,
                          payment_idx,
                          event_grid,
                          time_dependence,
                          int_dt)
        # Kappa on event grid.
        self.kappa_eg = self.zcbond.kappa_eg
        # Vol on event grid.
        self.vol_eg = self.zcbond.vol_eg
        # Discount curve on event grid.
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        # y-function on event grid.
        self.y_eg = self.zcbond.y_eg
        # v-function on event grid until expiry.
        self.v_eg_tmp = None
        self.v_eg = None
        # dv_dt-function on event grid until expiry.
        self.dv_dt_eg_tmp = None
        self.dv_dt_eg = None

        self.model = self.zcbond.model
        self.transformation = self.zcbond.transformation
        if option_type == "caplet":
            self.type = global_types.Instrument.CAPLET
        elif option_type == "floorlet":
            self.type = global_types.Instrument.FLOORLET
        else:
            raise ValueError(f"Unknown instrument type: {option_type}")

        self.initialization()

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    @property
    def fixing_event(self) -> float:
        return self.event_grid[self.fixing_idx]

    @property
    def payment_event(self) -> float:
        return self.event_grid[self.payment_idx]

    @property
    def tenor(self) -> float:
        return self.payment_event - self.fixing_event

    @property
    def fix_idx(self) -> int:
        return self.fixing_idx

    @fix_idx.setter
    def fix_idx(self, idx: int) -> None:
        self.fixing_idx = idx
        self.initialization()

    @property
    def pay_idx(self) -> int:
        return self.payment_idx

    @pay_idx.setter
    def pay_idx(self, idx: int) -> None:
        self.payment_idx = idx
        self.zcbond.mat_idx = idx
        self.update_v_function()

    def initialization(self) -> None:
        """Initialization of object."""
        if self.time_dependence == "constant":
            self.v_eg_tmp = \
                misc_ep.v_constant(self.zcbond.kappa_eg[0],
                                   self.zcbond.vol_eg[0],
                                   self.fixing_idx,
                                   self.event_grid)
            self.dv_dt_eg_tmp = \
                misc_ep.dv_dt_constant(self.zcbond.kappa_eg[0],
                                       self.zcbond.vol_eg[0],
                                       self.fixing_idx,
                                       self.event_grid)
        elif self.time_dependence == "piecewise":
            self.v_eg_tmp = \
                misc_ep.v_piecewise(self.zcbond.kappa_eg[0],
                                    self.zcbond.vol_eg,
                                    self.fixing_idx,
                                    self.event_grid)
            self.dv_dt_eg_tmp = \
                misc_ep.dv_dt_piecewise(self.zcbond.kappa_eg[0],
                                        self.zcbond.vol_eg,
                                        self.fixing_idx,
                                        self.event_grid)
        elif self.time_dependence == "general":
            self.v_eg_tmp = \
                misc_ep.v_general(self.zcbond.int_grid,
                                  self.zcbond.int_event_idx,
                                  self.zcbond.int_kappa_step_ig,
                                  self.zcbond.vol_ig,
                                  self.fixing_idx)
            self.dv_dt_eg_tmp = \
                misc_ep.dv_dt_general(self.zcbond.int_event_idx,
                                      self.zcbond.int_kappa_step_ig,
                                      self.zcbond.vol_ig,
                                      self.fixing_idx)
        else:
            raise ValueError(
                f"Unknown time dependence: {self.time_dependence}")
        self.update_v_function()

    def update_v_function(self) -> None:
        """Update v- and dv_dt-function."""
        # v-function on event grid until expiry.
        self.v_eg = \
            misc_ep.v_function(self.fixing_idx,
                               self.payment_idx,
                               self.zcbond.g_eg,
                               self.v_eg_tmp)
        # dv_dt-function on event grid until expiry.
        self.dv_dt_eg = \
            misc_ep.v_function(self.fixing_idx,
                               self.payment_idx,
                               self.zcbond.g_eg,
                               self.dv_dt_eg_tmp)

    def payoff(self,
               spot: typing.Union[float, np.ndarray],
               discounting: bool = False) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot pseudo short rate.
            discounting: Do analytical discounting from payment date to
                fixing date. Default is false.

        Returns:
            Payoff.
        """
        # P(t_fix, t_pay).
        price = self.zcbond_price(spot, self.fix_idx, self.pay_idx)
        # Simple rate at t_fix for (t_fix, t_pay).
        simple_rate = misc_sw.simple_forward_rate(price, self.tenor)
        # Payoff.
        if self.type == global_types.Instrument.CAPLET:
            _payoff = self.tenor * payoffs.call(simple_rate, self.strike_rate)
        else:
            _payoff = self.tenor * payoffs.put(simple_rate, self.strike_rate)
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
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return misc_cf.caplet_price(spot, self.strike_rate, self.tenor,
                                    event_idx, self.fix_idx, self.pay_idx,
                                    self.zcbond, self.v_eg, self.type)

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        return misc_cf.caplet_delta(spot, self.strike_rate, self.tenor,
                                    event_idx, self.fix_idx, self.pay_idx,
                                    self.zcbond, self.v_eg, self.type)

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        return misc_cf.caplet_gamma(spot, self.strike_rate, self.tenor,
                                    event_idx, self.fix_idx, self.pay_idx,
                                    self.zcbond, self.v_eg, self.type)

    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        return misc_cf.caplet_theta(spot, self.strike_rate, self.tenor,
                                    event_idx, self.fix_idx, self.pay_idx,
                                    self.zcbond, self.v_eg, self.dv_dt_eg,
                                    self.type)

    def fd_solve(self) -> None:
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Update drift, diffusion and rate vectors.
        self.fd_update(self.event_grid.size - 1)
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for counter, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - counter
            # Update drift, diffusion and rate vectors at previous event.
            self.fd_update(event_idx - 1)
            # Payoff at payment event, discounted to fixing event.
            if event_idx == self.fix_idx:
                self.fd.solution += self.payoff(self.fd.grid, True)
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjust_discount_steps[event_idx]

    def zcbond_price(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Price of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond price.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.price(spot, event_idx)

    def zcbond_delta(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Delta of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond delta.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.delta(spot, event_idx)

    def zcbond_gamma(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Gamma of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond gamma.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.gamma(spot, event_idx)

    def zcbond_theta(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Theta of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond delta.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.theta(spot, event_idx)

    def mc_exact_setup(self) -> None:
        """Setup exact Monte-Carlo solver."""
        self.zcbond.mc_exact_setup()
        self.mc_exact = self.zcbond.mc_exact

    def mc_exact_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        present_value = self.mc_present_value(self.mc_exact)
        self.mc_exact.mc_estimate = present_value.mean()
        self.mc_exact.mc_error = present_value.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self) -> None:
        """Setup Euler Monte-Carlo solver."""
        self.zcbond.mc_euler_setup()
        self.mc_euler = self.zcbond.mc_euler

    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        present_value = self.mc_present_value(self.mc_euler)
        self.mc_euler.mc_estimate = present_value.mean()
        self.mc_euler.mc_error = present_value.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)

    def mc_present_value(self,
                         mc_object) -> np.ndarray:
        """Present value for each Monte-Carlo path."""
        # Adjustment of discount paths.
        discount_paths = \
            mc_object.discount_adjustment(mc_object.discount_paths,
                                          self.adjust_discount)
        # Pseudo short rate at fixing event.
        spot = mc_object.rate_paths[self.fix_idx]
        # Option payoff at fixing event.
        option_payoff = self.payoff(spot, True)
        # Option payoff discounted back to present time.
        option_payoff *= discount_paths[self.fix_idx]
        return option_payoff


class CapletPelsser(Caplet):
    """Caplet or floorlet in 1-factor Hull-White model.

    Price of caplet of floorlet.

    See Pelsser (2000), Chapter 5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike_rate: Caplet or floorlet rate.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        option_type: Caplet or floorlet. Default is caplet.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike_rate: float,
                 fixing_idx: int,
                 payment_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 option_type: str = "caplet"):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         strike_rate,
                         fixing_idx,
                         payment_idx,
                         event_grid,
                         time_dependence,
                         int_dt,
                         option_type)

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondPelsser(kappa,
                                 vol,
                                 discount_curve,
                                 fixing_idx,
                                 event_grid,
                                 time_dependence,
                                 int_dt)

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount
