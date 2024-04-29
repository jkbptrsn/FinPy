import typing
import math

import numpy as np
from scipy.optimize import brentq

from models import options
from models.hull_white import european_option as option
from models.hull_white import misc_swap as misc_sw
from models.hull_white import swap
from utils import data_types
from utils import global_types
from utils import payoffs


class Payer(options.Option1FAnalytical):
    """European payer swaption in 1-factor Hull-White model.

    Price of European payer swaption based on a fixed-for-floating swap
    (based on "simple rate" fixing).

    See Andersen & Piterbarg (2010), Sections 5.10 and 10.1.3.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        fixed_rate: Fixed rate.
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
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 fixed_rate: float,
                 fixing_schedule: np.ndarray,
                 payment_schedule: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.fixed_rate = fixed_rate
        self.fixing_schedule = fixing_schedule
        self.payment_schedule = payment_schedule
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Underlying fixed-for-floating Swap.
        self.swap = \
            swap.Swap(kappa,
                      vol,
                      discount_curve,
                      fixed_rate,
                      fixing_schedule,
                      payment_schedule,
                      event_grid,
                      time_dependence,
                      int_dt)
        # Zero-coupon bond.
        self.zcbond = self.swap.zcbond

        # Put option written on zero-coupon bond.
        self.put = \
            option.EuropeanOption(kappa,
                                  vol,
                                  discount_curve,
                                  0,
                                  fixing_schedule[0],
                                  payment_schedule[-1],
                                  event_grid,
                                  time_dependence,
                                  int_dt,
                                  "Put")

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
        self.type = global_types.Instrument.SWAPTION

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    @property
    def expiry(self) -> float:
        return self.event_grid[self.fixing_schedule[0]]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.payment_schedule[-1]]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot value of underlying fixed-for-floating swap.

        Returns:
            Payoff.
        """
        return payoffs.call(spot, 0)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        swaption_price = 0
        # Pseudo short rate corresponding to zero swap value.
        expiry_idx = self.fixing_schedule[0]
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(expiry_idx,))
        for fix_idx, pay_idx in \
                zip(self.fixing_schedule, self.payment_schedule):
            # "Strike" of put option.
            self.zcbond.mat_idx = pay_idx
            self.put.strike = self.zcbond.price(rate_star, expiry_idx)
            self.put.mat_idx = pay_idx
            put_price = self.put.price(spot, event_idx)
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swaption_price += self.fixed_rate * tenor * put_price
            if pay_idx == self.event_grid.size - 1:
                swaption_price += put_price
        return swaption_price

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
        swaption_delta = 0
        # Pseudo short rate corresponding to zero swap value.
        expiry_idx = self.fixing_schedule[0]
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(expiry_idx,))
        for fix_idx, pay_idx in \
                zip(self.fixing_schedule, self.payment_schedule):
            # "Strike" of put option.
            self.zcbond.mat_idx = pay_idx
            self.put.strike = self.zcbond.price(rate_star, expiry_idx)
            self.put.mat_idx = pay_idx
            put_delta = self.put.delta(spot, event_idx)
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swaption_delta += self.fixed_rate * tenor * put_delta
            if pay_idx == self.event_grid.size - 1:
                swaption_delta += put_delta
        return swaption_delta

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
        swaption_gamma = 0
        # Pseudo short rate corresponding to zero swap value.
        expiry_idx = self.fixing_schedule[0]
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(expiry_idx,))
        for fix_idx, pay_idx in \
                zip(self.fixing_schedule, self.payment_schedule):
            # "Strike" of put option.
            self.zcbond.mat_idx = pay_idx
            self.put.strike = self.zcbond.price(rate_star, expiry_idx)
            self.put.mat_idx = pay_idx
            put_gamma = self.put.gamma(spot, event_idx)
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swaption_gamma += self.fixed_rate * tenor * put_gamma
            if pay_idx == self.event_grid.size - 1:
                swaption_gamma += put_gamma
        return swaption_gamma

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
        swaption_theta = 0
        # Pseudo short rate corresponding to zero swap value.
        expiry_idx = self.fixing_schedule[0]
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(expiry_idx,))
        for fix_idx, pay_idx in \
                zip(self.fixing_schedule, self.payment_schedule):
            # "Strike" of put option.
            self.zcbond.mat_idx = pay_idx
            self.put.strike = self.zcbond.price(rate_star, expiry_idx)
            self.put.mat_idx = pay_idx
            put_theta = self.put.theta(spot, event_idx)
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swaption_theta += self.fixed_rate * tenor * put_theta
            if pay_idx == self.event_grid.size - 1:
                swaption_theta += put_theta
        return swaption_theta

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
            # Swap payments.
            if event_idx in self.fixing_schedule:
                idx_fix = event_idx
                which_fix = np.where(self.fixing_schedule == idx_fix)
                idx_pay = self.payment_schedule[which_fix][0]
                # P(t_fixing, t_payment).
                bond_price = \
                    self.swap.zcbond_price(self.fd.grid, idx_fix, idx_pay)
                # Tenor.
                tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
                # Simple rate at t_fixing for (t_fixing, t_payment).
                simple_rate = misc_sw.simple_forward_rate(bond_price, tenor)
                # Payment.
                payment = tenor * (simple_rate - self.fixed_rate)
                # Analytical discounting from payment date to fixing date.
                payment *= bond_price
                self.fd.solution += payment
            # Option payoff.
            if event_idx == self.fixing_schedule[0]:
                self.fd.solution = self.payoff(self.fd.solution)
            # Propagation for one time step.
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
        # Pseudo short rate at expiry.
        expiry_idx = self.fixing_schedule[0]
        spot = mc_object.rate_paths[expiry_idx]
        # Swap price at expiry.
        swap_price = self.swap.price(spot, expiry_idx)
        # Option payoff at expiry.
        option_payoff = self.payoff(swap_price)
        # Option payoff discounted back to present time.
        option_payoff *= discount_paths[expiry_idx]
        return option_payoff


class PayerPelsser(Payer):
    """European payer swaption in 1-factor Hull-White model.

    Price of European payer swaption based on a fixed-for-floating swap
    (based on "simple rate" fixing).

    See Pelsser (2000), Chapter 5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        fixed_rate: Fixed rate.
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
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 fixed_rate: float,
                 fixing_schedule: np.ndarray,
                 payment_schedule: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         fixed_rate,
                         fixing_schedule,
                         payment_schedule,
                         event_grid,
                         time_dependence,
                         int_dt)

        # Underlying fixed-for-floating Swap.
        self.swap = \
            swap.SwapPelsser(kappa,
                             vol,
                             discount_curve,
                             fixed_rate,
                             fixing_schedule,
                             payment_schedule,
                             event_grid,
                             time_dependence,
                             int_dt)
        # Zero-coupon bond.
        self.zcbond = self.swap.zcbond

        # Put option written on zero-coupon bond.
        self.put = \
            option.EuropeanOptionPelsser(kappa,
                                         vol,
                                         discount_curve,
                                         0,
                                         fixing_schedule[0],
                                         payment_schedule[-1],
                                         event_grid,
                                         time_dependence,
                                         int_dt,
                                         "Put")

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount
