import typing

import numpy as np

from models import options
from models.hull_white import caplet_floorlet as xlet
from utils import data_types
from utils import global_types


class CapFloor(options.Option1FAnalytical):
    """Cap or floor in 1-factor Hull-White model.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike_rate: Cap or floor rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        cap_or_floor: Caplet or floorlet. Default is caplet.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 strike_rate: float,
                 fixing_schedule: np.ndarray,
                 payment_schedule: np.ndarray,
                 event_grid: np.ndarray,
                 cap_or_floor: str = "cap",
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike_rate = strike_rate
        self.fixing_schedule = fixing_schedule
        self.payment_schedule = payment_schedule
        self.event_grid = event_grid
        self.cap_or_floor = cap_or_floor
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
        # v-function on event grid.
        self.v_eg = None

        self.model = global_types.Model.HULL_WHITE_1F
        self.transformation = global_types.Transformation.ANDERSEN
        if self.cap_or_floor == "cap":
            self.type = global_types.Instrument.CAP
            caplet_floorlet = "caplet"
        elif self.cap_or_floor == "floor":
            self.type = global_types.Instrument.FLOOR
            caplet_floorlet = "floorlet"
        else:
            raise ValueError(f"Unknown instrument type: {self.cap_or_floor}")

        # Caplet/floorlet object used in analytical pricing.
        self.xlet = \
            xlet.Caplet(kappa, vol, discount_curve, strike_rate,
                        fixing_schedule[0], payment_schedule[0],
                        event_grid, time_dependence, int_step_size,
                        caplet_floorlet)

        self.initialization()

    # TODO: Expiry corresponds actually to the payment date.
    #  Maybe a new base call for options?
    @property
    def expiry(self) -> float:
        return self.event_grid[self.fixing_schedule[0]]

    def initialization(self):
        """Initialization of instrument object."""
        self.kappa_eg = self.xlet.kappa_eg
        self.vol_eg = self.xlet.vol_eg
        self.discount_curve_eg = self.xlet.discount_curve_eg
        self.forward_rate_eg = self.xlet.forward_rate_eg
        self.y_eg = self.xlet.y_eg
        self.v_eg = self.xlet.v_eg

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
        return 0 * spot

    def xlet_payoff(self,
                    spot: typing.Union[float, np.ndarray],
                    fixing_idx: int,
                    payment_idx: int) -> typing.Union[float, np.ndarray]:
        """..."""
        self.xlet.fixing_idx = fixing_idx
        self.xlet.payment_idx = payment_idx
        self.xlet.initialization()
        return self.xlet.payoff(spot, discounting=True)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        _price = 0
        # Assuming that event_idx <= self.fixing_schedule[0]
        for idx_fix, idx_pay in zip(self.fixing_schedule, self.payment_schedule):
            self.xlet.fixing_idx = idx_fix
            self.xlet.payment_idx = idx_pay
            self.xlet.initialization()
            _price += self.xlet.price(spot, event_idx)
        return _price

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
        _delta = 0
        # Assuming that event_idx <= self.fixing_schedule[0]
        for idx_fix, idx_pay in zip(self.fixing_schedule, self.payment_schedule):
            self.xlet.fixing_idx = idx_fix
            self.xlet.payment_idx = idx_pay
            self.xlet.initialization()
            _delta += self.xlet.delta(spot, event_idx)
        return _delta

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

            # Caplet/floorlet payoff at payment event, discount to
            # fixing event.
            if event_idx in self.fixing_schedule:
                idx_fix = event_idx
                which_fix = np.where(self.fixing_schedule == idx_fix)
                idx_pay = self.payment_schedule[which_fix][0]
                self.fd.solution += \
                    self.xlet_payoff(self.fd.grid, idx_fix, idx_pay)

            # Propagation for one time step dt.
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
