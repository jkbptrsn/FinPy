import typing

import numpy as np
from scipy.optimize import brentq

from models import options
from models.hull_white import misc as misc_hw
from models.hull_white import put_option
from models.hull_white import sde
from models.hull_white import swap
from models.hull_white import zero_coupon_bond
from models.hull_white import zero_coupon_bond as zcbond
from utils import global_types
from utils import misc
from utils import payoffs


class Payer(sde.SDE, options.VanillaOption):
    """European payer swaption class for the 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        expiry_idx: Event-grid index corresponding to expiry.
        maturity_idx: Event-grid index corresponding to maturity.
        fixed_rate: Fixed rate.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 expiry_idx: int,
                 maturity_idx: int,
                 fixed_rate: float,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
#        self.discount_curve = discount_curve
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.fixed_rate = fixed_rate

        self.option_type = global_types.Instrument.SWAPTION

        # Underlying swap
        # -- assuming swap maturity is equal to event_grid[-1]
        self.swap = swap.Swap(kappa, vol, discount_curve, event_grid,
                              fixed_rate, int_step_size=int_step_size)

        # Zero-coupon bond object with maturity at last event.
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    event_grid[-1],
                                    int_step_size=int_step_size)

        # Put option written on zero-coupon bond.
        self.put = put_option.Put(kappa, vol, discount_curve, event_grid,
                                  1, expiry_idx, maturity_idx,
                                  int_step_size=int_step_size)

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        pass

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See Eq. (10.24), L.B.G. Andersen & V.V. Piterbarg 2010.

        Args:
            spot: Spot value of PSEUDO short rate.
            event_idx: Event-grid index of current time.

        Returns:
            Swaption price.
        """
        swaption_price = 0
        # Pseudo short rate corresponding to zero swap value.
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(event_idx,))
        for mat_idx in range(event_idx + 1, self.event_grid.size):
            # "Strike" of put option
            self.zcbond.maturity_idx = mat_idx
            self.put.strike = self.zcbond.price(rate_star, event_idx)
            # Maturity of put option
            self.put.maturity_idx = mat_idx
            put_price = self.put.price(spot, event_idx)
            # Time between two adjacent payments in year fractions.
            tau = self.event_grid[mat_idx] - self.event_grid[mat_idx - 1]
            # ...
            swaption_price += self.fixed_rate * tau * put_price
            if mat_idx == self.event_grid.size - 1:
                swaption_price += put_price
        return swaption_price

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass


class PayerNew(options.EuropeanOptionAnalytical1F):
    """European payer swaption in 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        fixed_rate: Fixed rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
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
        self.option_type = global_types.Instrument.SWAPTION

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

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondNew(kappa, vol, discount_curve, event_grid.size - 1,
                             event_grid, time_dependence, int_step_size)
        # Put option written on zero-coupon bond.
        self.put = \
            put_option.PutNew(kappa, vol, discount_curve, 1,
                              fixing_schedule[0], payment_schedule[-1],
                              event_grid, time_dependence, int_step_size)
        # Swap.
        self.swap = \
            swap.SwapNew(kappa, vol, discount_curve, fixed_rate,
                         fixing_schedule, payment_schedule, event_grid,
                         time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.SWAPTION

    @property
    def expiry(self) -> float:
        return self.event_grid[self.fixing_schedule[0]]

    @property
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
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current value of underlying zero-coupon bond.

        Returns:
            Payoff.
        """
        return 0 * spot

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """

        swaption_price = 0
        # Pseudo short rate corresponding to zero swap value.
        expiry_idx = self.fixing_schedule[0]
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(expiry_idx,))

#        print(rate_star)

        for fix_idx, pay_idx in \
                zip(self.fixing_schedule, self.payment_schedule):

            # "Strike" of put option
            self.zcbond.maturity_idx = pay_idx
#            self.zcbond.initialization()
            self.put.strike = self.zcbond.price(rate_star, event_idx)

            # Maturity of put option
            self.put.maturity_idx = pay_idx
#            self.put.initialization()
            put_price = self.put.price(spot, event_idx)

            # Time between two adjacent payments in year fractions.
            tau = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # ...
            swaption_price += self.fixed_rate * tau * put_price

            if pay_idx == self.event_grid.size - 1:
                swaption_price += put_price

        return swaption_price

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
        swaption_delta = 0
        # Pseudo short rate corresponding to zero swap value.
        expiry_idx = self.fixing_schedule[0]
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(expiry_idx,))

        #        print(rate_star)

        for fix_idx, pay_idx in \
                zip(self.fixing_schedule, self.payment_schedule):

            # "Strike" of put option
            self.zcbond.maturity_idx = pay_idx
            #            self.zcbond.initialization()
            self.put.strike = self.zcbond.price(rate_star, event_idx)

            # Maturity of put option
            self.put.maturity_idx = pay_idx
            #            self.put.initialization()
            put_delta = self.put.delta(spot, event_idx)

            # Time between two adjacent payments in year fractions.
            tau = self.event_grid[pay_idx] - self.event_grid[fix_idx]

            # ...
            swaption_delta += self.fixed_rate * tau * put_delta

            if pay_idx == self.event_grid.size - 1:
                swaption_delta += put_delta

        return swaption_delta

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
                bond_price = \
                    self.swap.zcbond_price(self.fd.grid, idx_fix, idx_pay)
                # Tenor.
                tenor = self.event_grid[idx_pay] - self.event_grid[idx_fix]
                # Simple forward rate at t_fixing for (t_fixing, t_payment).
                simple_rate = \
                    self.swap.simple_forward_rate(bond_price, tenor)
                # Payment.
                payment = tenor * (simple_rate - self.fixed_rate)
                # Analytical discounting from payment date to fixing date.
                payment *= bond_price
                self.fd.solution += payment

            if event_idx == self.fixing_schedule[0]:
                self.fd.solution = payoffs.call(self.fd.solution, 0)

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
