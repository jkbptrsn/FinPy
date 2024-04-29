import math
import typing

import numpy as np

from models import bonds
from models.hull_white import zero_coupon_bond as zcbond
from utils import data_types
from utils import global_types


class Bond(bonds.Bond1FAnalytical):
    """Coupon bearing bond in 1-factor Hull-White model.

    Bond with pre-specified cash flow.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        cash_flow_schedule: Cash flow indices on event grid.
        cash_flow: Cash flow.
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
                 cash_flow_schedule: np.ndarray,
                 cash_flow: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.cash_flow_schedule = cash_flow_schedule
        self.cash_flow = cash_flow
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBond(kappa,
                          vol,
                          discount_curve,
                          self.cash_flow_schedule[-1],
                          event_grid,
                          time_dependence,
                          int_dt)
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
        self.type = global_types.Instrument.BOND

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

    def maturity(self) -> float:
        return self.event_grid[self.cash_flow_schedule[-1]]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot pseudo short rate.

        Returns:
            Payoff.
        """
        return self.cash_flow[-1] + 0 * spot

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Bond price.
        """
        _price = 0
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # Discount factor.
            discount = self.zcbond_price(spot, event_idx, idx_pay)
            _price += discount * self.cash_flow[count]
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
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # 1st order derivative of discount factor wrt short rate.
            discount = self.zcbond_delta(spot, event_idx, idx_pay)
            _delta += discount * self.cash_flow[count]
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
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # 2nd order derivative of discount factor wrt short rate.
            discount = self.zcbond_gamma(spot, event_idx, idx_pay)
            _gamma += discount * self.cash_flow[count]
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
        for count, idx_pay in enumerate(self.cash_flow_schedule):
            # 1st order derivative of discount factor wrt time.
            discount = self.zcbond_theta(spot, event_idx, idx_pay)
            _theta += discount * self.cash_flow[count]
        return _theta

    def fd_solve(self) -> None:
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Update drift, diffusion and rate vectors.
        self.fd_update(self.event_grid.size - 1)
        # Backwards propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for counter, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - counter
            # Update drift, diffusion and rate vectors at previous event.
            self.fd_update(event_idx - 1)
            # Payment at cash flow event.
            if event_idx in self.cash_flow_schedule:
                which_pay = np.where(self.cash_flow_schedule == event_idx)[0]
                self.fd.solution += self.cash_flow[which_pay[0]]
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjust_discount_steps[event_idx]

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
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
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
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
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
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.gamma(spot, event_idx)

    def zcbond_theta(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Theta of zero-coupon bond.

        Args:
            spot: Current value of pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond theta.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.theta(spot, event_idx)

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
        discount_paths = mc_object.discount_adjustment(
            mc_object.discount_paths, self.adjust_discount)
        bond_payoff = np.zeros(mc_object.discount_paths.shape[1])
        for counter, idx_pay in enumerate(self.cash_flow_schedule):
            bond_payoff += self.cash_flow[counter] * discount_paths[idx_pay]
        return bond_payoff


class BondPelsser(Bond):
    """Coupon bearing bond in 1-factor Hull-White model.

    Bond with pre-specified cash flow.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        cash_flow_schedule: Cash flow indices on event grid.
        cash_flow: Cash flow.
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
                 cash_flow_schedule: np.ndarray,
                 cash_flow: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         cash_flow_schedule,
                         cash_flow,
                         event_grid,
                         time_dependence,
                         int_dt)

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondPelsser(kappa,
                                 vol,
                                 discount_curve,
                                 self.cash_flow_schedule[-1],
                                 event_grid,
                                 time_dependence,
                                 int_dt)

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount
