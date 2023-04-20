import typing

import numpy as np

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


class SwapNew:
    """Fixed-for-floating swap in 1-factor Hull-White model.

    Fixed-for-floating swap based on LIBOR type rate fixing. Priced from
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

        # Underlying zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondNew(kappa, vol, discount_curve, event_grid.size - 1,
                             event_grid, time_dependence, int_step_size)

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.SWAP

    def initialization(self):
        """Initialization of instrument object."""
        self.kappa_eg = self.zcbond.kappa_eg
        self.vol_eg = self.zcbond.vol_eg
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        self.y_eg = self.zcbond.y_eg

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: TODO...

        Returns:
            Payoff.
        """
        pass

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

        # TODO: Test pricing function -- keep this version (extending to delta and gamma is easy)
        # TODO: Test annuity and forward_swap_rate functions
        # TODO: Calculate par rate

        self.update_remaining(event_idx)

#        # Remaining fixings.
#        fixing_remaining = \
#            self.fixing_schedule[self.fixing_schedule >= event_idx]
#        # Remaining payments.
#        payment_remaining = self.payment_schedule[-fixing_remaining.size:]

        swap_price = 0
        for fix_idx, pay_idx in \
                zip(self.fixing_remaining, self.payment_remaining):
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
        pass

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
        # Remaining fixings.
        self.fixing_remaining = \
            self.fixing_schedule[self.fixing_schedule >= event_idx]
        # Remaining payments.
        self.payment_remaining = \
            self.payment_schedule[-self.fixing_remaining.size:]

    def annuity(self,
                spot: (float, np.ndarray),
                event_idx: int) -> (float, np.ndarray):
        """Present Value of a Basis Point (PVBP).

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            PVBP.
        """

        self.update_remaining(event_idx)

        pvbp = 0
        for fix_idx, pay_idx in \
                zip(self.fixing_remaining, self.payment_remaining):
            # Price of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_price = self.zcbond.price(spot, event_idx)
            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]
            pvbp += tenor * bond_price
        return pvbp

    def forward_swap_rate(self,
                          spot: (float, np.ndarray),
                          event_idx: int) -> (float, np.ndarray):
        """Forward swap rate...

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            ...
        """

        self.update_remaining(event_idx)

        forward = 0
        for fix_idx, pay_idx in \
                zip(self.fixing_remaining, self.payment_remaining):

            # Price of zero-coupon bond maturing at fix_idx.
            self.zcbond.maturity_idx = fix_idx
            bond_price_fix = self.zcbond.price(spot, event_idx)
            # Price of zero-coupon bond maturing at pay_idx.
            self.zcbond.maturity_idx = pay_idx
            bond_price_pay = self.zcbond.price(spot, event_idx)

            # Tenor.
            tenor = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # "Libor rate".
            rate = (bond_price_fix - bond_price_pay) / (tenor * bond_price_pay)
            forward += tenor * bond_price_pay * rate

        return forward / self.annuity(spot, event_idx)
