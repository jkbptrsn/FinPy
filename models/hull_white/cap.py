import math
import typing

import numpy as np

from models import options
from models.hull_white import caplet
from utils import data_types
from utils import global_types


class Cap(options.Option1FAnalytical):
    """Cap or floor in 1-factor Hull-White model.

    Price of cap of floor.

    See Andersen & Piterbarg (2010), Proposition 4.5.2, and
    Brigo & Mercurio (2007), Section 3.3.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        strike_rate: Cap or floor rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        option_type: Cap or floor. Default is cap.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            strike_rate: float,
            fixing_schedule: np.ndarray,
            payment_schedule: np.ndarray,
            event_grid: np.ndarray,
            time_dependence: str = "piecewise",
            int_dt: float = 1 / 52,
            option_type: str = "cap"):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike_rate = strike_rate
        self.fixing_schedule = fixing_schedule
        self.payment_schedule = payment_schedule
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Caplet/floorlet.
        self.xlet = caplet.Caplet(
            kappa, vol, discount_curve, strike_rate, fixing_schedule[0],
            payment_schedule[0], event_grid, time_dependence, int_dt,
            option_type + "let")
        # Zero-coupon bond.
        self.zcbond = self.xlet.zcbond
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

        self.model = self.zcbond.model
        self.transformation = self.zcbond.transformation
        if option_type == "cap":
            self.type = global_types.Instrument.CAP
        elif option_type == "floor":
            self.type = global_types.Instrument.FLOOR
        else:
            raise ValueError(f"Unknown instrument type: {option_type}")

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    def payoff(
            self,
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
        return 0 * spot

    def xlet_payoff(self,
                    spot: typing.Union[float, np.ndarray],
                    fixing_idx: int,
                    payment_idx: int) -> typing.Union[float, np.ndarray]:
        """Payoff function for caplet or floorlet."""
        self.xlet.fix_idx = fixing_idx
        self.xlet.pay_idx = payment_idx
        return self.xlet.payoff(spot, discounting=True)

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
        _price = 0
        # Assuming that event_idx <= self.fixing_schedule[0]
        for idx_fix, idx_pay in zip(self.fixing_schedule, self.payment_schedule):
            self.xlet.fix_idx = idx_fix
            self.xlet.pay_idx = idx_pay
            _price += self.xlet.price(spot, event_idx)
        return _price

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
        _delta = 0
        # Assuming that event_idx <= self.fixing_schedule[0]
        for idx_fix, idx_pay in zip(self.fixing_schedule, self.payment_schedule):
            self.xlet.fix_idx = idx_fix
            self.xlet.pay_idx = idx_pay
            _delta += self.xlet.delta(spot, event_idx)
        return _delta

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
        _gamma = 0
        # Assuming that event_idx <= self.fixing_schedule[0]
        for idx_fix, idx_pay in zip(self.fixing_schedule, self.payment_schedule):
            self.xlet.fix_idx = idx_fix
            self.xlet.pay_idx = idx_pay
            _gamma += self.xlet.gamma(spot, event_idx)
        return _gamma

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
        _theta = 0
        # Assuming that event_idx <= self.fixing_schedule[0]
        for idx_fix, idx_pay in zip(self.fixing_schedule, self.payment_schedule):
            self.xlet.fix_idx = idx_fix
            self.xlet.pay_idx = idx_pay
            _theta += self.xlet.theta(spot, event_idx)
        return _theta

    def fd_solve(self) -> None:
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Update drift, diffusion and rate vectors.
        self.fd_update(self.event_grid.size - 1)
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for count, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - count
            # Update drift, diffusion and rate vectors at previous event.
            self.fd_update(event_idx - 1)
            # Payoff at payment event, discounted to fixing event.
            if event_idx in self.fixing_schedule:
                idx_fix = event_idx
                which_fix = np.where(self.fixing_schedule == idx_fix)
                idx_pay = self.payment_schedule[which_fix][0]
                self.fd.solution += \
                    self.xlet_payoff(self.fd.grid, idx_fix, idx_pay)
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjust_discount_steps[event_idx]

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

        Exact discretization.

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
        payoff = np.zeros(mc_object.discount_paths.shape[1])
        for idx_fix, idx_pay in \
                zip(self.fixing_schedule, self.payment_schedule):
            spot = mc_object.rate_paths[idx_fix]
            # Payoff discounted back to present time.
            payoff += self.xlet_payoff(spot, idx_fix, idx_pay) \
                * discount_paths[idx_fix]
        return payoff


class CapPelsser(Cap):
    """Cap or floor in 1-factor Hull-White model.

    Price of cap of floor.

    See Pelsser (2000), Chapter 5.

    See Andersen & Piterbarg (2010), Proposition 4.5.2, and
    Brigo & Mercurio (2007), Section 3.3.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        strike_rate: Cap or floor rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        option_type: Cap or floor. Default is cap.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike_rate: float,
                 fixing_schedule: np.ndarray,
                 payment_schedule: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 option_type: str = "cap"):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         strike_rate,
                         fixing_schedule,
                         payment_schedule,
                         event_grid,
                         time_dependence,
                         int_dt,
                         option_type)

        # Caplet/floorlet.
        self.xlet = \
            caplet.CapletPelsser(kappa,
                                 vol,
                                 discount_curve,
                                 strike_rate,
                                 fixing_schedule[0],
                                 payment_schedule[0],
                                 event_grid,
                                 time_dependence,
                                 int_dt,
                                 option_type + "let")
        # Zero-coupon bond.
        self.zcbond = self.xlet.zcbond

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount
