import typing

import numpy as np

from models import bonds
from models.hull_white import misc as misc_hw
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types
from utils import misc


class Swap(bonds.BondAnalytical1F):
    """Fixed-for-floating swap in 1-factor Hull-White model.

    Price of fixed-for-floating swap based on "simple rate" fixing.
    Priced from the point of view of the fixed rate payer.

    See L.B.G. Andersen & V.V. Piterbarg 2010, section 5.5.

    TODO: Call self.zcbond.mat_idx = "updated maturity index" before pricing

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.

        fixed_rate: Fixed rate.
        spread: Spread to floating rate. TODO!
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

        self.spread = None

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

        # Zero-coupon bond object used in analytical pricing.
        self.zcbond = \
            zcbond.ZCBond(kappa, vol, discount_curve, event_grid.size - 1,
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

    ####################################################################

    def maturity(self) -> float:
        return self.event_grid[self.payment_schedule[-1]]

    ####################################################################

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
        fix_idx = self.fixing_schedule[-1]
        pay_idx = self.payment_schedule[-1]
        # P(t_fixing, t_payment).
        bond_price = self.zcbond_price(spot, fix_idx, pay_idx)
        tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
        # Simple forward rate at t_fixing for (t_fixing, t_payment).
        simple_rate = self.simple_forward_rate(bond_price, tenor)
        _payoff = tenor * (simple_rate - self.fixed_rate)
        # Do analytical discounting from payment date to fixing date.
        if discounting:
            return _payoff * bond_price
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
            Swap price.
        """
        self.update_remaining(event_idx)
        swap_price = 0
        # Check if first payment has been fixed.
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            # Price of zero-coupon bond maturing at pay_idx.
            bond_price = self.zcbond_price(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
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
        """1st order price sensitivity wrt value of underlying.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        self.update_remaining(event_idx)
        swap_delta = 0
        # Check if first payment has been fixed.
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            # Delta of zero-coupon bond maturing at pay_idx.
            bond_delta = self.zcbond_delta(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
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
        """2nd order price sensitivity wrt value of underlying.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        self.update_remaining(event_idx)
        swap_gamma = 0
        # Check if first payment has been fixed.
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            # Gamma of zero-coupon bond maturing at pay_idx.
            bond_gamma = self.zcbond_gamma(spot, event_idx, pay_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
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
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Numerical propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for count, dt in enumerate(time_steps):
            # Event index before propagation over dt.
            event_idx = (self.event_grid.size - 1) - count

            # Update drift, diffusion, and rate functions.
            update_idx = event_idx - 1
            drift = \
                self.y_eg[update_idx] - self.kappa_eg[update_idx] * self.fd.grid
            diffusion = self.vol_eg[update_idx] + 0 * self.fd.grid
            rate = self.fd.grid + self.forward_rate_eg[update_idx]
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)

            # Payments, discount to fixing event.
            if event_idx in self.fixing_schedule:
                idx_fix = event_idx
                which_fix = np.where(self.fixing_schedule == idx_fix)
                idx_pay = self.payment_schedule[which_fix][0]
                # P(t_fixing, t_payment).
                bond_price = self.zcbond_price(self.fd.grid, idx_fix, idx_pay)
                # Tenor.
                tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
                # Simple forward rate at t_fixing for (t_fixing, t_payment).
                simple_rate = self.simple_forward_rate(bond_price, tenor)
                # Payment.
                payment = tenor * (simple_rate - self.fixed_rate)
                # Analytical discounting from payment date to fixing date.
                payment *= bond_price
                self.fd.solution += payment

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

    def mc_euler_setup(self):
        """Setup Euler Monte-Carlo solver."""
        pass

    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

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
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            PVBP.
        """
        self.update_remaining(event_idx)
        pvbp = 0
        # Check if first payment has been fixed.
        if self.slice_start == 1:
            pay_idx = self.payment_remaining[0]
            # Price of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_price = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
            pvbp = tenor * bond_price
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Price of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_price = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            pvbp += tenor * bond_price
        # Reset maturity index.
        self.zcbond.maturity_idx = self.event_grid.size - 1
        return pvbp

    def par_rate(self,
                 spot: typing.Union[float, np.ndarray],
                 event_idx: int,
                 floating_rate_fixed: float = 0) \
            -> typing.Union[float, np.ndarray]:
        """Calculate par rate, also referred to as forward swap rate.

        Args:
            spot: Current value of pseudo short rate.
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
            self.zcbond.maturity_idx = pay_idx
            bond_price = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
            forward_rate = tenor * bond_price * floating_rate_fixed
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Price of zero-coupon bond maturing at fix_idx.
            self.zcbond.maturity_idx = fix_idx
            bond_price_fix = self.zcbond.price(spot, event_idx)
            # Price of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_price_pay = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            # Simple forward_rate rate.
            rate = \
                self.simple_forward_rate(bond_price_pay, tenor, bond_price_fix)
            forward_rate += tenor * bond_price_pay * rate
        # Reset maturity index.
        self.zcbond.maturity_idx = self.event_grid.size - 1
        return forward_rate / self.annuity(spot, event_idx)

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

    def zcbond_gamma(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Gamma of zero-coupon bond.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond gamma.
        """
        if self.zcbond.maturity_idx != maturity_idx:
            self.zcbond.maturity_idx = maturity_idx
            self.zcbond.initialization()
        return self.zcbond.gamma(spot, event_idx)

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


class SwapPelsser(Swap):
    """Fixed-for-floating swap in 1-factor Hull-White model.

    Fixed-for-floating swap based on "simple rate" fixing. Priced from
    the point of view of the fixed rate payer. See Pelsser, chapter 5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        fixed_rate: Fixed rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_dt: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
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
                 int_dt: float = 1 / 365):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         fixed_rate,
                         fixing_schedule,
                         payment_schedule,
                         event_grid,
                         time_dependence,
                         int_dt)

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
                misc_hw.integration_grid(self.event_grid, self.int_dt)
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
        self.adjustment_rate = self.forward_rate_eg + alpha
        self.adjustment_discount = discount_steps * np.exp(-int_alpha)

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Numerical propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for count, dt in enumerate(time_steps):
            # Event index before propagation over dt.
            event_idx = (self.event_grid.size - 1) - count
            # Update drift, diffusion, and rate functions.
            update_idx = event_idx - 1
            drift = -self.kappa_eg[update_idx] * self.fd.grid
            diffusion = self.vol_eg[update_idx] + 0 * self.fd.grid
            rate = self.fd.grid
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)
            # Payments, discount to fixing event.
            if event_idx in self.fixing_schedule:
                idx_fix = event_idx
                which_fix = np.where(self.fixing_schedule == idx_fix)
                idx_pay = self.payment_schedule[which_fix][0]

                # P(t_fixing, t_payment).
                bond_price = self.zcbond_price(self.fd.grid, idx_fix, idx_pay)
#                grid = self.fd.grid + self.adjustment_rate[event_idx]
#                bond_price = self.zcbond_price(grid, idx_fix, idx_pay)

                # Tenor.
                tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
                # Simple forward rate at t_fixing for (t_fixing, t_payment).
                simple_rate = self.simple_forward_rate(bond_price, tenor)
                # Payment.
                payment = tenor * (simple_rate - self.fixed_rate)

                # Analytical discounting from payment date to fixing date.
                payment *= bond_price
#                payment *= np.prod(self.adjustment_discount[idx_fix + 1: idx_pay + 1])

                self.fd.solution += payment

            # Propagation for one time step.
            self.fd.propagation(dt, True)

            # Transformation adjustment.
            self.fd.solution *= self.adjustment_discount[event_idx]
