import typing

import math
import numpy as np

from models import bonds
from models.hull_white import misc as misc_hw
from models.hull_white import sde
from utils import global_types
from utils import misc
from utils import payoffs


class ZCBond(sde.SDE, bonds.VanillaBond):
    """Zero-coupon bond in 1-factor Hull-White model.

    TODO: Use rate or pseudo rate?

    Zero-coupon bond dependent on pseudo short rate modelled by 1-factor
    Hull-White SDE. See L.B.G. Andersen & V.V. Piterbarg 2010,
    proposition 10.1.7.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.

        discount_curve: Discount curve represented on event grid.

        maturity_idx: Maturity index on event grid.

        event_grid: Event dates represented as year fractions from as-of
            date.

        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 maturity_idx: int,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.discount_curve = discount_curve
        self.maturity_idx = maturity_idx

        self.bond_type = global_types.Instrument.ZERO_COUPON_BOND

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) -> \
            typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current short rate.

        Returns:
            Payoff.
        """
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return self._calc_price(spot, event_idx, self.maturity_idx)

    def price_vector(self,
                     spot: typing.Union[float, np.ndarray],
                     event_idx: int,
                     maturity_indices: np.ndarray) -> np.ndarray:
        """Price of zero coupon bond for each index in maturity_indices."""
        if isinstance(spot, np.ndarray):
            zcbond_prices = np.zeros((maturity_indices.size, spot.size))
        else:
            zcbond_prices = np.zeros((maturity_indices.size, 1))
        for idx, maturity_idx in enumerate(maturity_indices):
            zcbond_prices[idx] = self._calc_price(spot, event_idx, maturity_idx)
        return zcbond_prices

    def _calc_price(self,
                    spot: typing.Union[float, np.ndarray],
                    event_idx: int,
                    maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Calculate zero-coupon bond price.

        Calculate price of zero-coupon bond based at current time
        (event_idx) for maturity at time T (maturity_idx). See
        proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

        Args:
            spot: Current pseudo short rate.
            event_idx: Event grid index corresponding to current time.
            maturity_idx: Event grid index corresponding to maturity.

        Returns:
            Zero-coupon bond price.
        """
        if event_idx > maturity_idx:
            raise ValueError("event_idx > maturity_idx")

        # P(0,t): Zero-coupon bond price at time zero with maturity t.
        price1 = self.discount_curve.values[event_idx]
        # P(0,T): Zero-coupon bond price at time zero with maturity T.
        price2 = self.discount_curve.values[maturity_idx]

        # Integration indices of the two relevant events.
        int_idx1 = self.int_event_idx[event_idx]
        int_idx2 = self.int_event_idx[maturity_idx] + 1

        # Slice of integration grid.
        int_grid = self.int_grid[int_idx1:int_idx2]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = self.int_kappa_step[int_idx1:int_idx2]
        # G(t,T): G-function,
        # see Eq. (10.18), L.B.G. Andersen & V.V. Piterbarg 2010.
        integrand = np.exp(-np.cumsum(int_kappa))
        g = np.sum(misc.trapz(int_grid, integrand))
        # y(t): y-function,
        # see Eq. (10.17), L.B.G. Andersen & V.V. Piterbarg 2010.
        y = self.y_event_grid[event_idx]
        return price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1

###############################################################################

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
        pass

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
        pass

    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Current pseudo short rate.
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


class ZCBondNew(bonds.VanillaBondAnalytical1F):
    """Zero-coupon bond in 1-factor Hull-White model.

    TODO: Use rate or pseudo rate?

    Zero-coupon bond dependent on pseudo short rate modelled by 1-factor
    Hull-White SDE. See L.B.G. Andersen & V.V. Piterbarg 2010,
    proposition 10.1.7.

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
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_step_size = int_step_size

        # Speed of mean reversion on event grid.
        self.kappa_eg = None
        # Volatility on event grid.
        self.vol_eg = None
        # Discount curve on event grid.
        self.discount_curve_eg = None
        # G-function on event grid.
        self.g_eg = None
        # y-function on event grid.
        self.y_eg = None

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
        # G-function on integration grid.
        self.g_ig = None
        # y-function on integration grid.
        self.y_ig = None

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.ZERO_COUPON_BOND

        # Initialization
        self._setup_int_grid()
        self._setup_model_parameters()

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def _setup_int_grid(self):
        """Set up time grid for numerical integration."""
        self.int_grid, self.int_event_idx = \
            misc_hw.setup_int_grid(self.event_grid, self.int_step_size)

    def _setup_model_parameters(self):
        """Set up model parameters on event grid."""
        # Speed of mean reversion interpolated on event grid.
        self.kappa_eg = self.kappa.interpolation(self.event_grid)
        # Volatility interpolated on event grid.
        self.vol_eg = self.vol.interpolation(self.event_grid)
        # Discount curve interpolated on event grid.
        self.discount_curve_eg = \
            self.discount_curve.interpolation(self.event_grid)
        if self.time_dependence == "constant":
            # G-function on event grid.
            self.g_eg = misc_hw.g_constant(self.kappa_eg[0],
                                           self.maturity_idx,
                                           self.event_grid)
            # y-function on event grid.
            self.y_eg = misc_hw.y_constant(self.kappa_eg[0],
                                           self.vol_eg[0],
                                           self.event_grid)
        elif self.time_dependence == "piecewise":
            # G-function on event grid.
            self.g_eg = misc_hw.g_constant(self.kappa_eg[0],
                                           self.maturity_idx,
                                           self.event_grid)
            # y-function on event grid.
            self.y_eg = misc_hw.y_piecewise(self.kappa_eg[0],
                                            self.vol_eg,
                                            self.event_grid)
        elif self.time_dependence == "general":
            # Speed of mean reversion interpolated on integration grid.
            self.kappa_ig = self.kappa.interpolation(self.int_grid)
            # Volatility interpolated on integration grid.
            self.vol_ig = self.vol.interpolation(self.int_grid)
            # Integration of speed of mean reversion using trapezoidal rule.
            self.int_kappa_step = \
                np.append(0, misc.trapz(self.int_grid, self.kappa_ig))
            # G-function on event grid.
            self.g_eg, self.g_ig = misc_hw.g_general(self.int_grid,
                                                     self.int_event_idx,
                                                     self.int_kappa_step,
                                                     self.maturity_idx,
                                                     self.event_grid)
            # y-function on event and integration grid.
            self.y_eg, self.y_ig = misc_hw.y_general(self.int_grid,
                                                     self.int_event_idx,
                                                     self.int_kappa_step,
                                                     self.vol_ig,
                                                     self.event_grid)
        else:
            raise ValueError(f"Time dependence unknown: "
                             f"{self.time_dependence}")

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) -> \
            typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current short rate.

        Returns:
            Payoff.
        """
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current pseudo short rate.
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
            spot: Current pseudo short rate.
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
            spot: Current pseudo short rate.
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
            spot: Current pseudo short rate.
            event_idx: Index on event grid.
            type_: Calculation type: "price", "delta" or "gamma".
                Default is "price".

        Returns:
            Zero-coupon bond price/delta/gamma.
        """
        if event_idx > self.maturity_idx:
            raise ValueError("event_idx > maturity_idx")
        # P(0,t): Zero-coupon bond price at time zero with maturity t.
        price1 = self.discount_curve_eg[event_idx]
        # P(0,T): Zero-coupon bond price at time zero with maturity T.
        price2 = self.discount_curve_eg[self.maturity_idx]
        # G(t,T): G-function.
        g = self.g_eg[event_idx]
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
            spot: Current pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

###############################################################################

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()

#        # Set terminal condition.
#        self.fd.solution = self.payoff(self.fd.grid)

        for count, dt in enumerate(np.flip(np.diff(self.event_grid))):

            idx = -2 - count
            drift = self.y_eg[idx] - self.kappa_eg[idx] * self.fd.grid
            diffusion = self.vol_eg[idx] + 0 * self.fd.grid
            self.fd.set_drift(drift)
            self.fd.set_diffusion(diffusion)

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
