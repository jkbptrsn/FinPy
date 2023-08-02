import math
import typing

import numpy as np

from models import options
from models.hull_white import misc as misc_hw
from models.hull_white import misc_european_option as misc_ep
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types

from utils.global_types import Instrument
from utils.global_types import Model

from utils import payoffs


class EuropeanOption(options.Option1FAnalytical):
    """European call/put option in 1-factor Hull-White model.

    Price of European call/put option written on zero-coupon bond. See
    L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike: Strike value of underlying zero-coupon bond.
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        option_type: Option type. Default is call.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 option_type: str = "Call"):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Underlying zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBond(kappa, vol, discount_curve, maturity_idx,
                          event_grid, time_dependence, int_dt)
        # Speed of mean reversion on event grid.
        self.kappa_eg = self.zcbond.kappa_eg
        # Volatility on event grid.
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

        self.model = Model.HULL_WHITE_1F
        self.transformation = global_types.Transformation.ANDERSEN
        if option_type == "Call":
            self.type = Instrument.EUROPEAN_CALL
        elif option_type == "Put":
            self.type = Instrument.EUROPEAN_PUT
        else:
            raise ValueError(f"Option type is unknown: {option_type}")

        self.initialization()

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.zcbond.maturity

    ####################################################################

    @property
    def exp_idx(self) -> int:
        return self.expiry_idx

    @exp_idx.setter
    def exp_idx(self, idx: int):
        self.expiry_idx = idx
        self.initialization()

    @property
    def mat_idx(self) -> int:
        return self.maturity_idx

    @mat_idx.setter
    def mat_idx(self, idx: int):
        self.maturity_idx = idx
        self.zcbond.maturity_idx = idx
        self.zcbond.gt_eg = \
            misc_hw.g_function(idx,
                               self.zcbond.g_eg,
                               self.zcbond.int_kappa_eg)
        self.update_v_function()

    ####################################################################

    def initialization(self):
        """Initialization of instrument object."""
        if self.time_dependence == "constant":
            self.v_eg_tmp = \
                misc_ep.v_constant(self.zcbond.kappa_eg[0],
                                   self.zcbond.vol_eg[0],
                                   self.expiry_idx,
                                   self.event_grid)
            self.dv_dt_eg_tmp = \
                misc_ep.dv_dt_constant(self.zcbond.kappa_eg[0],
                                       self.zcbond.vol_eg[0],
                                       self.expiry_idx,
                                       self.event_grid)
        elif self.time_dependence == "piecewise":
            self.v_eg_tmp = \
                misc_ep.v_piecewise(self.zcbond.kappa_eg[0],
                                    self.zcbond.vol_eg,
                                    self.expiry_idx,
                                    self.event_grid)
            self.dv_dt_eg_tmp = \
                misc_ep.dv_dt_piecewise(self.zcbond.kappa_eg[0],
                                        self.zcbond.vol_eg,
                                        self.expiry_idx,
                                        self.event_grid)
        elif self.time_dependence == "general":
            self.v_eg_tmp = \
                misc_ep.v_general(self.zcbond.int_grid,
                                  self.zcbond.int_event_idx,
                                  self.zcbond.int_kappa_step_ig,
                                  self.zcbond.vol_ig,
                                  self.expiry_idx)
            self.dv_dt_eg_tmp = \
                misc_ep.dv_dt_general(self.zcbond.int_event_idx,
                                      self.zcbond.int_kappa_step_ig,
                                      self.zcbond.vol_ig,
                                      self.expiry_idx)
        else:
            raise ValueError(f"Time dependence unknown: "
                             f"{self.time_dependence}")
        self.update_v_function()

    def update_v_function(self):
        """Update v- and dv_dt-function."""
        # v-function on event grid until expiry.
        self.v_eg = \
            misc_ep.v_function(self.expiry_idx,
                               self.maturity_idx,
                               self.zcbond.g_eg,
                               self.v_eg_tmp)
        # dv_dt-function on event grid until expiry.
        self.dv_dt_eg = \
            misc_ep.v_function(self.expiry_idx,
                               self.maturity_idx,
                               self.zcbond.g_eg,
                               self.dv_dt_eg_tmp)

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot value of underlying zero-coupon bond.

        Returns:
            Payoff.
        """
        if self.type == Instrument.EUROPEAN_CALL:
            return payoffs.call(spot, self.strike)
        else:
            return payoffs.put(spot, self.strike)

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
        return misc_ep.option_price(spot, self.strike, event_idx,
                                    self.expiry_idx, self.maturity_idx,
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
        return misc_ep.option_delta(spot, self.strike, event_idx,
                                    self.expiry_idx, self.maturity_idx,
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
        return misc_ep.option_gamma(spot, self.strike, event_idx,
                                    self.expiry_idx, self.maturity_idx,
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
        return misc_ep.option_theta(spot, self.strike, event_idx,
                                    self.expiry_idx, self.maturity_idx,
                                    self.zcbond, self.v_eg,
                                    self.dv_dt_eg, self.type)

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = self.zcbond.payoff(self.fd.grid)
        # Update drift, diffusion and rate vectors.
        self.fd_update(self.event_grid.size - 1)
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for counter, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - counter
            # Update drift, diffusion and rate vectors at previous event.
            self.fd_update(event_idx - 1)
            # Payoff at option expiry.
            if event_idx == self.expiry_idx:
                self.fd.solution = self.payoff(self.fd.solution)
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjust_discount_steps[event_idx]

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        self.zcbond.mc_exact_setup()
        self.mc_exact = self.zcbond.mc_exact

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
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        # Adjustment of discount paths.
        discount_paths = \
            self.mc_exact.discount_adjustment(self.mc_exact.discount_paths,
                                              self.adjust_discount)
        # Spot pseudo short rate at expiry.
        spot = self.mc_exact.rate_paths[self.expiry_idx]
        # Zero-coupon bond price at expiry.
        zcbond_price = self.zcbond.price(spot, self.expiry_idx)
        # Payoff at option expiry.
        option_payoff = self.payoff(zcbond_price)
        # Discounted option payoff.
        option_payoff *= discount_paths[self.expiry_idx]
        # Monte-Carlo estimate.
        self.mc_exact.mc_estimate = option_payoff.mean()
        self.mc_exact.mc_error = option_payoff.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self):
        """Setup Euler Monte-Carlo solver."""
        self.zcbond.mc_euler_setup()
        self.mc_euler = self.zcbond.mc_euler

    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Monte-Carlo paths constructed using Euler-Maruyama discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        # Adjustment of discount paths.
        discount_paths = \
            self.mc_euler.discount_adjustment(self.mc_euler.discount_paths,
                                              self.adjust_discount)
        # Spot pseudo short rate at expiry.
        spot = self.mc_euler.rate_paths[self.expiry_idx]
        # Zero-coupon bond price at expiry.
        zcbond_price = self.zcbond.price(spot, self.expiry_idx)
        # Option payoff at expiry.
        option_payoff = self.payoff(zcbond_price)
        # Discounted option payoff.
        option_payoff *= discount_paths[self.expiry_idx]
        # Monte-Carlo estimate.
        self.mc_euler.mc_estimate = option_payoff.mean()
        self.mc_euler.mc_error = option_payoff.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)


class EuropeanOptionPelsser(EuropeanOption):
    """European call/put option in 1-factor Hull-White model.

    Price of European call/put option written on zero-coupon bond. See
    A. Pelsser, chapter 5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike: Strike value of underlying zero-coupon bond.
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        option_type: Option type. Default is call.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 option_type: str = "Call"):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         strike,
                         expiry_idx,
                         maturity_idx,
                         event_grid,
                         time_dependence,
                         int_dt,
                         option_type)

        # Underlying zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondPelsser(kappa, vol, discount_curve, maturity_idx,
                                 event_grid, time_dependence, int_dt)
        # Speed of mean reversion on event grid.
        self.kappa_eg = self.zcbond.kappa_eg
        # Volatility on event grid.
        self.vol_eg = self.zcbond.vol_eg
        # Discount curve on event grid.
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        # y-function on event grid.
        self.y_eg = self.zcbond.y_eg

        self.transformation = global_types.Transformation.PELSSER

        self.initialization()

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

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
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        # Adjustment of discount paths.
        discount_paths = \
            self.mc_exact.discount_adjustment(self.mc_exact.discount_paths,
                                              self.adjust_discount)

        # TODO: Adjustment of pseudo short rate to pseudo short rate...
        # Spot pseudo short rate at expiry.
        spot = self.mc_exact.rate_paths[self.expiry_idx]
        spot += self.adjust_rate[self.expiry_idx] \
            - self.forward_rate_eg[self.expiry_idx]

        # Zero-coupon bond price at expiry.
        zcbond_price = self.zcbond.price(spot, self.expiry_idx)
        # Payoff at option expiry.
        option_payoff = self.payoff(zcbond_price)
        # Discounted option payoff.
        option_payoff *= discount_paths[self.expiry_idx]
        # Monte-Carlo estimate.
        self.mc_exact.mc_estimate = option_payoff.mean()
        self.mc_exact.mc_error = option_payoff.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Monte-Carlo paths constructed using Euler-Maruyama discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        # Adjustment of discount paths.
        discount_paths = \
            self.mc_euler.discount_adjustment(self.mc_euler.discount_paths,
                                              self.adjust_discount)

        # TODO: Adjustment of pseudo short rate to pseudo short rate...
        # Spot pseudo short rate at expiry.
        spot = self.mc_euler.rate_paths[self.expiry_idx]
        spot += self.adjust_rate[self.expiry_idx] \
            - self.forward_rate_eg[self.expiry_idx]

        # Zero-coupon bond price at expiry.
        zcbond_price = self.zcbond.price(spot, self.expiry_idx)
        # Option payoff at expiry.
        option_payoff = self.payoff(zcbond_price)
        # Discounted option payoff.
        option_payoff *= discount_paths[self.expiry_idx]
        # Monte-Carlo estimate.
        self.mc_euler.mc_estimate = option_payoff.mean()
        self.mc_euler.mc_error = option_payoff.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)
