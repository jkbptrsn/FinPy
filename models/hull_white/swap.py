import typing

import numpy as np

from models import bonds
from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from models.hull_white import zero_coupon_bond as zcbond
from utils import global_types
from utils import misc


class Swap(sde.SDE):
    """Fixed-for-floating swap for the 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Payment dates represented as year fractions from the
            as-of date.
        fixed_rate: Fixed rate.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 fixed_rate: float,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.fixed_rate = fixed_rate

        self.instrument_type = global_types.Instrument.SWAP

        # Zero-coupon bond object with maturity at last event.
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    event_grid[-1],
                                    int_step_size=int_step_size)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See section 5.3, L.B.G. Andersen & V.V. Piterbarg 2010.

        Returns:
            Swap price.
        """
        swap_price = 0
        # Remaining event grid.
        event_grid_tmp = self.event_grid[event_idx:]
        for idx, tau in enumerate(np.diff(event_grid_tmp)):
            # Price of zero-coupon bond maturing at idx.
            self.zcbond.maturity_idx = event_idx + idx
            bond_price = self.zcbond.price(spot, event_idx)
            swap_price += bond_price
            # Price of zero-coupon bond maturing at idx + 1.
            self.zcbond.maturity_idx = event_idx + idx + 1
            bond_price = self.zcbond.price(spot, event_idx)
            swap_price -= (1 + tau * self.fixed_rate) * bond_price
        return swap_price


class SwapNew(bonds.VanillaBondAnalytical1F):
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
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
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

        # Underlying zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondNew(kappa, vol, discount_curve, event_grid.size - 1,
                             event_grid, time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.SWAP

    def initialization(self):
        """Initialization of instrument object."""
        self.kappa_eg = self.zcbond.kappa_eg
        self.vol_eg = self.zcbond.vol_eg
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        self.y_eg = self.zcbond.y_eg

    def maturity(self) -> float:
        return self.event_grid[self.payment_schedule[-1]]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current value of pseudo short rate.

        Returns:
            Payoff.
        """
        # Price, at last fixing date, of zero-coupon bond maturing at
        # last payment date.
        fix_idx = self.fixing_schedule[-1]
        pay_idx = self.payment_schedule[-1]
        self.zcbond.maturity_idx = pay_idx
        bond_price = self.zcbond.price(spot, fix_idx)
        # Tenor.
        tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
        simple_rate = self.simple_forward_rate(bond_price, tenor)
        return tenor * (simple_rate - self.fixed_rate)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
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
            self.zcbond.maturity_idx = pay_idx
            bond_price = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
            swap_price = 1 - (1 + tenor * self.fixed_rate) * bond_price
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Price of zero-coupon bond maturing at fix_idx.
            self.zcbond.maturity_idx = fix_idx
            bond_price = self.zcbond.price(spot, event_idx)
            swap_price += bond_price
            # Price of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_price = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_price -= (1 + tenor * self.fixed_rate) * bond_price
        # Reset maturity index.
        self.zcbond.maturity_idx = self.event_grid.size - 1
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
            self.zcbond.maturity_idx = pay_idx
            bond_delta = self.zcbond.delta(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
            swap_delta = 1 - (1 + tenor * self.fixed_rate) * bond_delta
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Delta of zero-coupon bond maturing at fix_idx.
            self.zcbond.maturity_idx = fix_idx
            bond_delta = self.zcbond.delta(spot, event_idx)
            swap_delta += bond_delta
            # Delta of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_delta = self.zcbond.delta(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_delta -= (1 + tenor * self.fixed_rate) * bond_delta
        # Reset maturity index.
        self.zcbond.maturity_idx = self.event_grid.size - 1
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
            self.zcbond.maturity_idx = pay_idx
            bond_gamma = self.zcbond.gamma(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[event_idx]
            swap_gamma = 1 - (1 + tenor * self.fixed_rate) * bond_gamma
        for fix_idx, pay_idx in zip(self.fixing_remaining,
                                    self.payment_remaining[self.slice_start:]):
            # Gamma of zero-coupon bond maturing at fix_idx.
            self.zcbond.maturity_idx = fix_idx
            bond_gamma = self.zcbond.gamma(spot, event_idx)
            swap_gamma += bond_gamma
            # Gamma of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_gamma = self.zcbond.gamma(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            swap_gamma -= (1 + tenor * self.fixed_rate) * bond_gamma
        # Reset maturity index.
        self.zcbond.maturity_idx = self.event_grid.size - 1
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

        print(self.event_grid)

#        self.fd.solution = self.payoff(self.fd.grid)

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

            # Payments.
            pay_idx = self.event_grid.size - 1 - count
            if pay_idx in self.payment_schedule:
                fix_idx = \
                    self.fixing_schedule[self.fixing_schedule < pay_idx][-1]
                self.zcbond.maturity_idx = pay_idx
                bond_price = self.zcbond.price(self.fd.grid, fix_idx)
                # Tenor.
                tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]

                print(count, fix_idx, pay_idx, tenor)

                simple_rate = self.simple_forward_rate(bond_price, tenor)

                print(simple_rate)

                self.fd.solution += tenor * (simple_rate - self.fixed_rate)

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
                spot: (float, np.ndarray),
                event_idx: int) -> (float, np.ndarray):
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
                 spot: (float, np.ndarray),
                 event_idx: int,
                 floating_rate_fixed: float = 0) -> (float, np.ndarray):
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

    @staticmethod
    def simple_forward_rate(bond_price_t2: (float, np.ndarray),
                            tau: float,
                            bond_price_t1: (float, np.ndarray) = 1.0) \
            -> (float, np.ndarray):
        """Calculate simple forward_rate rate.

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
