import math
import typing

import numpy as np

from models import bonds
from models.hull_white import mc_andersen as mc_a
from models.hull_white import mc_pelsser as mc_p
from models.hull_white import misc as misc_hw
from utils import data_types
from utils import global_types
from utils import payoffs


class ZCBond(bonds.BondAnalytical1F):
    """Zero-coupon bond in 1-factor Hull-White model.

    Zero-coupon bond price dependent on pseudo short rate modelled by
    1-factor Hull-White SDE. See L.B.G. Andersen & V.V. Piterbarg 2010,
    proposition 10.1.7.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
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
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt

        # Kappa on event grid.
        self.kappa_eg = None
        # Vol on event grid.
        self.vol_eg = None
        # Discount curve on event grid.
        self.discount_curve_eg = None
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = None
        # Integration of kappa on event_grid.
        self.int_kappa_eg = None
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
        self.kappa_ig = None
        # Vol on integration grid.
        self.vol_ig = None
        # Step-wise integration of kappa on integration grid.
        self.int_kappa_step_ig = None

        self.model = global_types.Model.HULL_WHITE_1F
        self.transformation = global_types.Transformation.ANDERSEN
        self.type = global_types.Instrument.ZERO_COUPON_BOND

        self.initialization()

        self.adjustment_rate = None
        self.adjustment_discount = None
        self.adjustment_function()

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    ####################################################################

    @property
    def mat_idx(self) -> int:
        return self.maturity_idx

    @mat_idx.setter
    def mat_idx(self, idx: int):
        self.maturity_idx = idx
        self.gt_eg = misc_hw.g_function(idx, self.g_eg, self.int_kappa_eg)

    ####################################################################

    def initialization(self):
        """Initialization of object."""
        if self.time_dependence == "general":
            self._setup_int_grid()
        self._setup_model_parameters()

    def adjustment_function(self):
        """Adjustment of short rate transformation."""
        self.adjustment_rate = self.forward_rate_eg
        self.adjustment_discount = self.discount_curve_eg

    def _setup_int_grid(self):
        """Set up time grid for numerical integration."""
        self.int_grid, self.int_event_idx = \
            misc_hw.integration_grid(self.event_grid, self.int_dt)

    def _setup_model_parameters(self):
        """Set up model parameters on event and integration grids."""
        misc_hw.setup_model_parameters(self)
        # G-function, G(t,t_maturity), on event grid.
        self.gt_eg = misc_hw.g_function(self.maturity_idx,
                                        self.g_eg,
                                        self.int_kappa_eg)

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot pseudo short rate.

        Returns:
            Payoff.
        """
        return payoffs.zero_coupon_bond(spot)

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
        return self._price_delta_gamma(spot, event_idx, "price")

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
        return self._price_delta_gamma(spot, event_idx, "delta")

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
        return self._price_delta_gamma(spot, event_idx, "gamma")

    def _price_delta_gamma(self,
                           spot: typing.Union[float, np.ndarray],
                           event_idx: int,
                           type_: str = "price") \
            -> typing.Union[float, np.ndarray]:
        """Calculate zero-coupon bond price, delta or gamma.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            type_: Calculation type: "price", "delta" or "gamma".
                Default is "price".

        Returns:
            Zero-coupon bond price, delta or gamma.
        """
        if event_idx > self.maturity_idx:
            raise ValueError("event_idx > maturity_idx")
        # P(0,t): Zero-coupon bond price at time zero with maturity t.
        price1 = self.discount_curve_eg[event_idx]
        # P(0,T): Zero-coupon bond price at time zero with maturity T.
        price2 = self.discount_curve_eg[self.maturity_idx]
        # G(t,T): G-function.
        g = self.gt_eg[event_idx]
        # y(t): y-function.
        y = self.y_eg[event_idx]
        # P(t,T): Zero-coupon bond price at time t with maturity T.
        bond_price = price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1
        if type_ == "price":
            return bond_price
        elif type_ == "delta":
            return -g * bond_price
        elif type_ == "gamma":
            return g ** 2 * bond_price
        else:
            raise ValueError(f"Calculation type is unknown: {type_}")

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
        # G(t,T): G-function.
        g = self.gt_eg[event_idx]
        # dG(t,T) / dt.
        dg_dt = -1 + self.kappa_eg[event_idx] * g
        # y(t): y-function.
        y = self.y_eg[event_idx]
        # dy(t) / dt.
        dy_dt = self.vol_eg[event_idx] ** 2 - 2 * self.kappa_eg[event_idx] * y
        theta = self._price_delta_gamma(spot, event_idx, "price")
        theta *= (-spot * dg_dt - dy_dt * g ** 2 / 2 - y * g * dg_dt)
        return theta

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = self.payoff(self.fd.grid)
        # Update drift, diffusion and rate vectors.
        self.fd_update(self.event_grid.size - 1)
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - idx
            # Update drift, diffusion and rate vectors at previous event.
            self.fd_update(event_idx - 1)
            self.fd.propagation(dt, True)

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        if self.time_dependence == "constant":
            self.mc_exact = mc_a.SdeExactConstant(self.kappa,
                                                  self.vol,
                                                  self.discount_curve,
                                                  self.event_grid)
        elif self.time_dependence == "piecewise":
            self.mc_exact = mc_a.SdeExactPiecewise(self.kappa,
                                                   self.vol,
                                                   self.discount_curve,
                                                   self.event_grid)
        elif self.time_dependence == "general":
            self.mc_exact = mc_a.SdeExactGeneral(self.kappa,
                                                 self.vol,
                                                 self.discount_curve,
                                                 self.event_grid,
                                                 self.int_dt)
        else:
            raise ValueError(f"Time dependence is unknown: "
                             f"{self.time_dependence}")

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
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        # Adjustment of discount paths.
        discount_paths = \
            self.mc_exact.discount_adjustment(self.mc_exact.discount_paths,
                                              self.adjustment_discount)
        self.mc_exact.mc_estimate = discount_paths[-1].mean()
        self.mc_exact.mc_error = discount_paths[-1].std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self):
        """Setup Euler Monte-Carlo solver."""
        self.mc_euler = mc_a.SdeEuler(self.kappa,
                                      self.vol,
                                      self.discount_curve,
                                      self.event_grid,
                                      self.time_dependence,
                                      self.int_dt)

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
        # Adjustment of discount paths.
        discount_paths = \
            self.mc_euler.discount_adjustment(self.mc_euler.discount_paths,
                                              self.adjustment_discount)
        self.mc_euler.mc_estimate = discount_paths[-1].mean()
        self.mc_euler.mc_error = discount_paths[-1].std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)


class ZCBondPelsser(ZCBond):
    """Zero-coupon bond in 1-factor Hull-White model.

    Zero-coupon bond price dependent on pseudo short rate modelled by
    1-factor Hull-White SDE. See A. Pelsser 2000, chapter 5.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise constant.
            "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 365):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         maturity_idx,
                         event_grid,
                         time_dependence,
                         int_dt)

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
            alpha = \
                misc_hw.alpha_general(self.int_grid,
                                      self.int_event_idx,
                                      self.int_kappa_step_ig,
                                      self.vol_ig,
                                      self.event_grid)
            int_alpha = \
                misc_hw.int_alpha_general(self.int_grid,
                                          self.int_event_idx,
                                          self.int_kappa_step_ig,
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
        for count, dt in enumerate(np.flip(np.diff(self.event_grid))):
            # Event index before propagation with time step -dt.
            event_idx = (self.event_grid.size - 1) - count
            # Update drift, diffusion, and rate functions.
            idx = event_idx - 1
            drift = -self.kappa_eg[idx] * self.fd.grid
            diffusion = self.vol_eg[idx] + 0 * self.fd.grid
            rate = self.fd.grid
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)
            self.fd.set_rate(rate)
            # Propagation for one time step.
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjustment_discount[event_idx]

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        if self.time_dependence == "constant":
            self.mc_exact = mc_p.SdeExactConstant(self.kappa,
                                                  self.vol,
                                                  self.discount_curve,
                                                  self.event_grid)
        elif self.time_dependence == "piecewise":
            self.mc_exact = mc_p.SdeExactPiecewise(self.kappa,
                                                   self.vol,
                                                   self.discount_curve,
                                                   self.event_grid)
        elif self.time_dependence == "general":
            self.mc_exact = mc_p.SdeExactGeneral(self.kappa,
                                                 self.vol,
                                                 self.discount_curve,
                                                 self.event_grid,
                                                 self.int_dt)
        else:
            raise ValueError(f"Time-dependence is unknown: "
                             f"{self.time_dependence}")

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
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        # Adjustment of discount paths.
        tmp = np.cumprod(self.adjustment_discount)
        discount_paths = \
            self.mc_exact.discount_adjustment(self.mc_exact.discount_paths,
                                              tmp)
        self.mc_exact.mc_estimate = discount_paths[-1].mean()
        self.mc_exact.mc_error = discount_paths[-1].std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)
