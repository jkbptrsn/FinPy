import typing

import numpy as np

from models import bonds
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types


class Swap(bonds.VanillaBondAnalytical1F):
    """Fixed-for-floating swap in 1-factor Hull-White model.

    Fixed-for-floating swap based on "simple rate" fixing. Priced from
    the point of view of the fixed rate payer. See
    L.B.G. Andersen & V.V. Piterbarg 2010, section 5.5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        fixed_rate: Fixed rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
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
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.fixed_rate = fixed_rate
        self.fixing_schedule = fixing_schedule
        self.payment_schedule = payment_schedule
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

        # Remaining fixing dates.
        self.fixing_remaining = None
        # Remaining payment dates.
        self.payment_remaining = None
        # First index for slicing of remaining payment dates.
        self.slice_start = None

        # Zero-coupon bond object used in analytical pricing.
        self.zcbond = \
            zcbond.ZCBond(kappa, vol, discount_curve, event_grid.size - 1,
                          event_grid, time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.transformation = global_types.Transformation.ANDERSEN
        self.type = global_types.Instrument.SWAP

    def maturity(self) -> float:
        return self.event_grid[self.payment_schedule[-1]]

    def initialization(self):
        """Initialization of instrument object."""
        self.kappa_eg = self.zcbond.kappa_eg
        self.vol_eg = self.zcbond.vol_eg
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        self.y_eg = self.zcbond.y_eg

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
            drift = \
                self.y_eg[event_idx] - self.kappa_eg[event_idx] * self.fd.grid
            diffusion = self.vol_eg[event_idx] + 0 * self.fd.grid
            rate = self.fd.grid + self.forward_rate_eg[event_idx]
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
