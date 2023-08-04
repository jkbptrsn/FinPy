import math
import typing

import numpy as np

from models import bonds
from models.hull_white import misc_swap as misc_sw
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types


class Swap(bonds.BondAnalytical1F):
    """Fixed-for-floating swap in 1-factor Hull-White model.

    Price of fixed-for-floating swap based on "simple rate" fixing.
    Priced from the point of view of the fixed rate payer.

    See L.B.G. Andersen & V.V. Piterbarg 2010, section 5.5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        fixed_rate: Fixed rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
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

        # Remaining fixing dates.
        self.fixing_remaining = None
        # Remaining payment dates.
        self.payment_remaining = None
        # First index for slicing of remaining payment dates.
        self.slice_start = None

        # Zero-coupon bond used in analytical pricing.
        self.zcbond = \
            zcbond.ZCBond(kappa, vol, discount_curve,
                          event_grid.size - 1,
                          event_grid, time_dependence, int_dt)
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
        self.type = global_types.Instrument.SWAP

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    def maturity(self) -> float:
        return self.event_grid[self.payment_schedule[-1]]

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
        fix_idx = self.fixing_schedule[-1]
        pay_idx = self.payment_schedule[-1]
        # P(t_fixing, t_payment).
        bond_price = self.zcbond_price(spot, fix_idx, pay_idx)
        tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
        # Simple rate at t_fixing for (t_fixing, t_payment).
        simple_rate = misc_sw.simple_forward_rate(bond_price, tenor)
        _payoff = tenor * (simple_rate - self.fixed_rate)
        # Analytical discounting from payment date to fixing date.
        if discounting:
            return _payoff * bond_price
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
            Swap price.
        """
        self.update_remaining(event_idx)
        swap_price = 0
        # Check if first payment has been fixed. Assume corresponding
        # fixing event is represented on event grid!
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            which_idx = np.where(self.payment_schedule == pay_idx)
            fix_idx = self.fixing_schedule[which_idx][0]
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # TODO: Is P(event_idx, fix_idx) = 1, when event_idx > fix_idx?
            swap_price = 1 - (1 + tenor * self.fixed_rate) * bond_price

        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Price of zero-coupon bond maturing at fix_idx.
            bond_price = self.zcbond_price(spot, event_idx, fix_idx)
            swap_price += bond_price
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_price -= (1 + tenor * self.fixed_rate) * bond_price
        return swap_price

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
        self.update_remaining(event_idx)
        swap_delta = 0
        # Check if first payment has been fixed. Assume corresponding
        # fixing event is represented on event grid!
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            which_idx = np.where(self.payment_schedule == pay_idx)
            fix_idx = self.fixing_schedule[which_idx][0]
            # Delta of zero-coupon bond maturing at pay_idx.
            bond_delta = self.zcbond_delta(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # TODO: Is delta of P(event_idx, fix_idx) = 1, when event_idx > fix_idx?
            swap_delta = 1 - (1 + tenor * self.fixed_rate) * bond_delta

        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Delta of zero-coupon bond maturing at fix_idx.
            bond_delta = self.zcbond_delta(spot, event_idx, fix_idx)
            swap_delta += bond_delta
            # Delta of zero-coupon bond maturing at pay_idx.
            bond_delta = self.zcbond_delta(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_delta -= (1 + tenor * self.fixed_rate) * bond_delta
        return swap_delta

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
        self.update_remaining(event_idx)
        swap_gamma = 0
        # Check if first payment has been fixed. Assume corresponding
        # fixing event is represented on event grid!
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            which_idx = np.where(self.payment_schedule == pay_idx)
            fix_idx = self.fixing_schedule[which_idx][0]
            # Gamma of zero-coupon bond maturing at pay_idx.
            bond_gamma = self.zcbond_gamma(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # TODO: Is delta of P(event_idx, fix_idx) = 1, when event_idx > fix_idx?
            swap_gamma = 1 - (1 + tenor * self.fixed_rate) * bond_gamma

        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Gamma of zero-coupon bond maturing at fix_idx.
            bond_gamma = self.zcbond_gamma(spot, event_idx, fix_idx)
            swap_gamma += bond_gamma
            # Gamma of zero-coupon bond maturing at pay_idx.
            bond_gamma = self.zcbond_gamma(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_gamma -= (1 + tenor * self.fixed_rate) * bond_gamma
        return swap_gamma

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
        self.update_remaining(event_idx)
        swap_theta = 0
        # Check if first payment has been fixed. Assume corresponding
        # fixing event is represented on event grid!
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            which_idx = np.where(self.payment_schedule == pay_idx)
            fix_idx = self.fixing_schedule[which_idx][0]
            # Gamma of zero-coupon bond maturing at pay_idx.
            bond_theta = self.zcbond_theta(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # TODO: Is delta of P(event_idx, fix_idx) = 1, when event_idx > fix_idx?
            swap_theta = 1 - (1 + tenor * self.fixed_rate) * bond_theta

        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Gamma of zero-coupon bond maturing at fix_idx.
            bond_theta = self.zcbond_theta(spot, event_idx, fix_idx)
            swap_theta += bond_theta
            # Gamma of zero-coupon bond maturing at pay_idx.
            bond_theta = self.zcbond_theta(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_theta -= (1 + tenor * self.fixed_rate) * bond_theta
        return swap_theta

    def fd_solve(self):
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
            # Payments.
            if event_idx in self.fixing_schedule:
                idx_fix = event_idx
                which_fix = np.where(self.fixing_schedule == idx_fix)
                idx_pay = self.payment_schedule[which_fix][0]
                # P(t_fixing, t_payment).
                bond_price = self.zcbond_price(self.fd.grid, idx_fix, idx_pay)
                # Tenor.
                tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
                # Simple rate at t_fixing for (t_fixing, t_payment).
                simple_rate = misc_sw.simple_forward_rate(bond_price, tenor)
                # Payment.
                payment = tenor * (simple_rate - self.fixed_rate)
                # Analytical discounting from payment date to fixing date.
                payment *= bond_price
                self.fd.solution += payment
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
        present_value = self.mc_present_value(self.mc_exact)
        self.mc_exact.mc_estimate = present_value.mean()
        self.mc_exact.mc_error = present_value.std(ddof=1)
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
        swap_payoff = np.zeros(mc_object.discount_paths.shape[1])
        for idx_fix, idx_pay in \
                zip(self.fixing_schedule, self.payment_schedule):
            # Pseudo short rate at fixing.
            spot = mc_object.rate_paths[idx_fix]
            # P(t_fixing, t_payment).
            bond_price = self.zcbond_price(spot, idx_fix, idx_pay)
            # Tenor.
            tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
            # Simple rate at t_fixing for (t_fixing, t_payment).
            simple_rate = misc_sw.simple_forward_rate(bond_price, tenor)
            # Payment.
            payment = tenor * (simple_rate - self.fixed_rate)
            # Discounting from payment date to present time.
            payment *= discount_paths[idx_pay]
            swap_payoff += payment
        return swap_payoff

    def update_remaining(self,
                         event_idx: int):
        """Update remaining fixing and payment dates.

        Args:
            event_idx: Index on event grid.
        """
        # Remaining fixing dates.
        self.fixing_remaining = \
            self.fixing_schedule[self.fixing_schedule >= event_idx]
        # Remaining payment dates.
        self.payment_remaining = \
            self.payment_schedule[self.payment_schedule >= event_idx]
        # First index for slicing of remaining payment dates.
        if self.fixing_remaining.size < self.payment_remaining.size:
            self.slice_start = 1
        else:
            self.slice_start = 0

    def annuity(self,
                spot: typing.Union[float, np.ndarray],
                event_idx: int) -> typing.Union[float, np.ndarray]:
        """Calculate Present Value of a Basis Point (PVBP).

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            PVBP.
        """
        self.update_remaining(event_idx)
        pvbp = 0
        # Check if first payment has been fixed. Assume corresponding
        # fixing event is represented on event grid!
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            which_idx = np.where(self.payment_schedule == pay_idx)
            fix_idx = self.fixing_schedule[which_idx][0]
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            pvbp = tenor * bond_price
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            pvbp += tenor * bond_price
        return pvbp

    def par_rate(self,
                 spot: typing.Union[float, np.ndarray],
                 event_idx: int,
                 floating_rate_fixed: float = 0) \
            -> typing.Union[float, np.ndarray]:
        """Calculate par rate, also referred to as forward swap rate.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            floating_rate_fixed: Floating rate fixed for first payment
                date, if event_idx is larger than first fixing index.

        Returns:
            Par rate.
        """
        self.update_remaining(event_idx)
        forward_rate = 0
        # Check if first payment has been fixed.
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
            forward_rate = tenor * bond_price * floating_rate_fixed
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Price of zero-coupon bond maturing at fix_idx.
            bond_price_fix = self.zcbond_price(spot, event_idx, fix_idx)
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price_pay = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            # Simple forward_rate rate.
            rate = misc_sw.simple_forward_rate(bond_price_pay, tenor,
                                               bond_price_fix)
            forward_rate += tenor * bond_price_pay * rate
        return forward_rate / self.annuity(spot, event_idx)

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


class SwapPelsser(Swap):
    """Fixed-for-floating swap in 1-factor Hull-White model.

    Price of fixed-for-floating swap based on "simple rate" fixing.
    Priced from the point of view of the fixed rate payer.

    See A. Pelsser, chapter 5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        fixed_rate: Fixed rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
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

        # Underlying zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondPelsser(kappa, vol, discount_curve,
                                 event_grid.size - 1,
                                 event_grid, time_dependence, int_dt)

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    def mc_present_value(self,
                         mc_object):
        """Present value for each Monte-Carlo path."""
        # Adjustment of discount paths.
        discount_paths = \
            mc_object.discount_adjustment(mc_object.discount_paths,
                                          self.adjust_discount)
        swap_payoff = np.zeros(mc_object.discount_paths.shape[1])
        for idx_fix, idx_pay in \
                zip(self.fixing_schedule, self.payment_schedule):
            # Pseudo short rate (Andersen transformation) at expiry.
            spot = mc_object.rate_paths[idx_fix]
            spot += self.adjust_rate[idx_fix] - self.forward_rate_eg[idx_fix]
            # P(t_fixing, t_payment).
            bond_price = self.zcbond_price(spot, idx_fix, idx_pay)
            # Tenor.
            tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
            # Simple rate at t_fixing for (t_fixing, t_payment).
            simple_rate = misc_sw.simple_forward_rate(bond_price, tenor)
            # Payment.
            payment = tenor * (simple_rate - self.fixed_rate)
            # Discounting from payment date to present time.
            payment *= discount_paths[idx_pay]
            swap_payoff += payment
        return swap_payoff
