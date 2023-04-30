import typing

import numpy as np

from models import bonds
from models.hull_white import zero_coupon_bond as zcbond
from utils import global_types
from utils import misc


class Bond(bonds.VanillaBondAnalytical1F):
    """Bond in 1-factor Hull-White model.

    Bond with pre-specified cash flow.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        cash_flow_schedule: Cash flow indices on event grid.
        cash_flow: Cash flow.
        event_grid: Event dates represented as year fractions from as-of
            date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise constant.
            "general": General time dependence.
            Default is "piecewise".
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 cash_flow_schedule: np.ndarray,
                 cash_flow: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.cash_flow_schedule = cash_flow_schedule
        self.cash_flow = cash_flow
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_step_size = int_step_size

        # Speed of mean reversion on event grid.
        self.kappa_eg = None
        # Volatility on event grid.
        self.vol_eg = None
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = None
        # y-function on event grid.
        self.y_eg = None

        # Zero-coupon bond object.
        self.zcbond = \
            zcbond.ZCBondNew(kappa, vol, discount_curve,
                             self.cash_flow_schedule[-1],
                             event_grid, time_dependence, int_step_size)

        self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.BOND

    def maturity(self) -> float:
        return self.event_grid[self.cash_flow_schedule[-1]]

    def initialization(self):
        """Initialization of instrument object."""
        self.kappa_eg = self.zcbond.kappa_eg
        self.vol_eg = self.zcbond.vol_eg
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        self.y_eg = self.zcbond.y_eg

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current pseudo short rate.

        Returns:
            Payoff.
        """
        return self.cash_flow[-1] + 0 * spot

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Bond price.
        """
        _price = 0
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # Discount factor from idx_pay to event_idx.
            discount = self.zcbond_price(spot, event_idx, idx_pay)
            _price += discount * self.cash_flow[count]
        return _price

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Current pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        _delta = 0
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # 1st order derivative of discount factor from idx_pay to
            # event_idx.
            discount = self.zcbond_delta(spot, event_idx, idx_pay)
            _delta += discount * self.cash_flow[count]
        return _delta

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Current pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        _gamma = 0
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # 2nd order derivative of discount factor from idx_pay to
            # event_idx.
            discount = self.zcbond_gamma(spot, event_idx, idx_pay)
            _gamma += discount * self.cash_flow[count]
        return _gamma

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
        # Backwards propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for count, dt in enumerate(time_steps):
            # Event index before backwards propagation over dt.
            event_idx = (self.event_grid.size - 1) - count
            # Update drift, diffusion, and rate functions.
            drift = \
                self.y_eg[event_idx] - self.kappa_eg[event_idx] * self.fd.grid
            diffusion = self.vol_eg[event_idx] + 0 * self.fd.grid
            rate = self.fd.grid + self.forward_rate_eg[event_idx]
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)
            # Cash flow events.
            if event_idx in self.cash_flow_schedule:
                which_payment = \
                    np.where(self.cash_flow_schedule == event_idx)[0]
                self.fd.solution += self.cash_flow[which_payment[0]]
            # Backwards propagation over dt.
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
