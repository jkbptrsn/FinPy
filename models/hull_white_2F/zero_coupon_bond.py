import math
import typing

import numpy as np

from models import bonds
from utils import data_types
from utils import global_types
from utils import payoffs


class ZCBond(bonds.BondAnalytical2F):
    """Zero-coupon bond in 2-factor Hull-White model.

    Price of zero-coupon bond dependent on pseudo short rate modelled by
    2-factor Hull-White SDE.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 12.1.4.

    Attributes:
        kappa_x: Speed of mean reversion in x-dimension.
        kappa_y: Speed of mean reversion in y-dimension.
        vol_x: Volatility in x-dimension.
        vol_y: Volatility in y-dimension.
        discount_curve: Discount curve represented on event grid.
        maturity_idx: Maturity index on event grid.
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
                 kappa_x: data_types.DiscreteFunc,
                 kappa_y: data_types.DiscreteFunc,
                 vol_x: data_types.DiscreteFunc,
                 vol_y: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__()
        self.kappa_x = kappa_x
        self.kappa_y = kappa_y
        self.vol_x = vol_x
        self.vol_y = vol_y
        self.discount_curve = discount_curve
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Kappa on event grid.
        self.kappa_x_eg = None
        self.kappa_y_eg = None
        # Vol on event grid.
        self.vol_x_eg = None
        self.vol_y_eg = None
        # Discount curve on event grid.
        self.discount_curve_eg = None
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = None
        # Integration of kappa on event_grid.
        self.int_kappa_x_eg = None
        self.int_kappa_y_eg = None
        # G-function, G(0,t), on event grid.
        self.g_eg = None
        # G-function, G(t,t_maturity), on event grid.
        self.gt_eg = None
        # y-function on event grid.
        self.y_eg = None

        # Integration grid.
        self.int_grid = None
        # Indices of event dates on integration grid.
        self.int_event_idx = None
        # Kappa on integration grid.
        self.kappa_x_ig = None
        self.kappa_y_ig = None
        # Vol on integration grid.
        self.vol_x_ig = None
        self.vol_y_ig = None
        # Step-wise integration of kappa on integration grid.
        self.int_kappa_x_step_ig = None
        self.int_kappa_y_step_ig = None

        self.model = global_types.Model.HULL_WHITE_2F
        self.transformation = global_types.Transformation.ANDERSEN
        self.type = global_types.Instrument.ZERO_COUPON_BOND

        self.adjust_rate = None
        self.adjust_discount_steps = None
        self.adjust_discount = None

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot pseudo short rate.

        Returns:
            Payoff.
        """
        pass

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        pass

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
        pass

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
        pass

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

        Monte-Carlo paths constructed using exact discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
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

        Monte-Carlo paths constructed using Euler-Maruyama discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        pass

    def mc_present_value(self,
                         mc_object):
        """Present value for each Monte-Carlo path."""
        pass
