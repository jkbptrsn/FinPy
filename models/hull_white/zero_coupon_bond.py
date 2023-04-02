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

        # Functions on event grid.
        self.kappa_eg = None
        self.vol_eg = None
        self.discount_curve_eg = None
        self.y_eg = None

        # Integration grid.
        self.int_grid = None
        # Indices of event dates on integration grid.
        self.int_event_idx = None
        # y-function on integration grid.
        self.y_int_grid = None

        # Array only used in initialization of the SDE object.
        self.int_kappa_step = None

        self.model = global_types.Model.HULL_WHITE_1F
        self.type = global_types.Instrument.ZERO_COUPON_BOND

        self._setup_int_grid()
        self._setup_model_parameters()

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def _setup_int_grid(self):
        """Construct time grid for numerical integration."""
        # Assume that the first event is the initial time point on the
        # integration grid.
        self.int_grid = np.array(self.event_grid[0])
        # The first event has index zero on the integration grid.
        self.int_event_idx = np.array(0)
        # Step size between two adjacent events.
        step_size_grid = np.diff(self.event_grid)
        for idx, step_size in enumerate(step_size_grid):
            # Number of integration steps.
            steps = math.floor(step_size / self.int_step_size)
            initial_date = self.event_grid[idx]
            if steps == 0:
                grid = np.array(initial_date + step_size)
            else:
                grid = self.int_step_size * np.arange(1, steps + 1) \
                    + initial_date
                diff_step = step_size - steps * self.int_step_size
                if diff_step > 1.0e-8:
                    grid = np.append(grid, grid[-1] + diff_step)
            self.int_grid = np.append(self.int_grid, grid)
            self.int_event_idx = np.append(self.int_event_idx, grid.size)
        self.int_event_idx = np.cumsum(self.int_event_idx)

    def _setup_model_parameters(self):
        """Set up model parameters on event grid."""
        self.kappa_eg = self.kappa.interpolation(self.event_grid)
        self.vol_eg = self.vol.interpolation(self.event_grid)
        self.discount_curve_eg = \
            self.discount_curve.interpolation(self.event_grid)
        if self.time_dependence == "constant":
            self.y_eg = y_constant(self.kappa_eg[0],
                                   self.vol_eg[0],
                                   self.event_grid)
        elif self.time_dependence == "piecewise":
            self.y_eg = y_piecewise(self.kappa_eg[0],
                                    self.vol_eg,
                                    self.event_grid)
        elif self.time_dependence == "general":
            self.y_eg = y_general(self.kappa,
                                  self.vol,
                                  self.event_grid,
                                  self.int_grid,
                                  self.int_event_idx,
                                  self.int_kappa_step,
                                  self.y_int_grid)
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
        if self.time_dependence == "constant":
            return zcbond_constant(spot,
                                   self.kappa_eg[0],
                                   self.vol_eg[0],
                                   self.discount_curve_eg,
                                   event_idx,
                                   self.maturity_idx,
                                   self.event_grid,
                                   self.y_eg,
                                   type_)
        elif self.time_dependence == "piecewise":
            return zcbond_piecewise(spot,
                                    self.kappa_eg[0],
                                    self.vol_eg,
                                    self.discount_curve_eg,
                                    event_idx,
                                    self.maturity_idx,
                                    self.event_grid,
                                    self.y_eg,
                                    type_)
        elif self.time_dependence == "general":
            return zcbond_general(spot,
                                  self.discount_curve_eg,
                                  event_idx,
                                  self.maturity_idx,
                                  self.int_event_idx,
                                  self.int_grid,
                                  self.int_kappa_step,
                                  self.y_eg,
                                  type_)
        else:
            raise ValueError(f"Time dependence unknown: "
                             f"{self.time_dependence}")

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
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = self.payoff(self.fd.grid)
        for count, dt in enumerate(np.flip(np.diff(self.event_grid))):

#            idx = -2 - count
#            drift = self.y_eg[idx] - self.kappa_eg[idx] * self.fd.grid
#            diffusion = self.vol_eg[idx] + 0 * self.fd.grid
#            self.fd.set_drift(drift)
#            self.fd.set_diffusion(diffusion)

            self.fd.propagation(dt)

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


def y_constant(kappa: float,
               vol: float,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function.

    Constant kappa and vol.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        y-function.
    """
    two_kappa = 2 * kappa
    return vol ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa


def y_piecewise(kappa: float,
                vol: np.ndarray,
                event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function.

    Constant kappa and piecewise constant vol.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        y-function.
    """
    y_return = np.zeros(event_grid.size)
    two_kappa = 2 * kappa
    for idx in range(1, event_grid.size):
        vol_times = event_grid[event_grid <= event_grid[idx]]
        vol_values = vol[event_grid <= event_grid[idx]]
        delta_t = event_grid[idx] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa
        y_return[idx] = y.sum()
    return y_return


def y_general(kappa: misc.DiscreteFunc,
              vol: misc.DiscreteFunc,
              event_grid: np.ndarray,
              int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step: np.ndarray,
              y_int_grid: np.ndarray):
    """Calculate y-function.

    General time-dependence of kappa and vol.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: ...
        int_grid: ...
        int_event_idx: ...
        int_kappa_step: ...
        y_int_grid: ...

    Returns:
        y-function.
    """
    # Speed of mean reversion interpolated on integration grid.
    kappa_int_grid = kappa.interpolation(int_grid)
    # Volatility interpolated on integration grid.
    vol_int_grid = vol.interpolation(int_grid)
    # Integration of speed of mean reversion using trapezoidal rule.
    int_kappa_step = np.append(0, misc.trapz(int_grid, kappa_int_grid))
    # Calculation of y-function on integration grid.
    y_int_grid = np.zeros(int_grid.size)
    for idx in range(1, int_grid.size):
        # int_u^t_{idx} kappa_s ds.
        int_kappa = int_kappa_step[:idx + 1]
        int_kappa = np.cumsum(int_kappa[::-1])[::-1]
        int_kappa[:-1] = int_kappa[1:]
        int_kappa[-1] = 0
        # Integrand in expression for y.
        integrand = np.exp(-2 * int_kappa) * vol_int_grid[:idx + 1] ** 2
        y_int_grid[idx] = np.sum(misc.trapz(int_grid[:idx + 1], integrand))
    # Save y-function on event grid.
    y_return = np.zeros(event_grid.size)
    for idx, event_idx in enumerate(int_event_idx):
        y_return[idx] = y_int_grid[event_idx]
    return y_return


def zcbond_constant(spot: typing.Union[float, np.ndarray],
                    kappa: float,
                    vol: float,
                    discount_curve: np.ndarray,
                    event_idx: int,
                    maturity_idx: int,
                    event_grid: np.ndarray,
                    y_eg: np.ndarray,
                    type_: str = "price") -> typing.Union[float, np.ndarray]:
    """Calculate zero-coupon bond price, delta or gamma.

    Calculate price/delta/gamma of zero-coupon bond based at current
    time (event_idx) for maturity at time T (maturity_idx).

    Assuming that speed of mean reversion and volatility are constant.

    Args:
        spot: Current pseudo short rate.
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        event_idx: Event grid index corresponding to current time.
        maturity_idx: Event grid index corresponding to maturity.
        event_grid: Event dates represented as year fractions from as-of
            date.
        y_eg: ...
        type_: Calculation type: "price", "delta" or "gamma".
            Default is "price".

    Returns:
        Zero-coupon bond price/delta/gamma.
    """
    if event_idx > maturity_idx:
        raise ValueError("event_idx > maturity_idx")
    # P(0,t): Zero-coupon bond price at time zero with maturity t.
    price1 = discount_curve[event_idx]
    # P(0,T): Zero-coupon bond price at time zero with maturity T.
    price2 = discount_curve[maturity_idx]
    # G(t,T): G-function,
    delta_t = event_grid[maturity_idx] - event_grid[event_idx]
    g = (1 - math.exp(-kappa * delta_t)) / kappa
    # y(t): y-function,
    y = y_eg[event_idx]
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    price = price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1
    if type_ == "price":
        return price
    elif type_ == "delta":
        return -g * price
    elif type_ == "gamma":
        return g ** 2 * price
    else:
        raise ValueError(f"Calculation type is unknown: {type_}")


def zcbond_piecewise(spot: typing.Union[float, np.ndarray],
                     kappa: float,
                     vol: np.ndarray,
                     discount_curve: np.ndarray,
                     event_idx: int,
                     maturity_idx: int,
                     event_grid: np.ndarray,
                     y_eg: np.ndarray,
                     type_: str = "price") -> typing.Union[float, np.ndarray]:
    """Calculate zero-coupon bond price, delta or gamma.

    Calculate price/delta/gamma of zero-coupon bond based at current
    time (event_idx) for maturity at time T (maturity_idx). See
    proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    Assuming that speed of mean reversion is constant and volatility is
    piecewise constant.

    Args:
        spot: Current pseudo short rate.
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        event_idx: Event grid index corresponding to current time.
        maturity_idx: Event grid index corresponding to maturity.
        event_grid: Event dates represented as year fractions from as-of
            date.
        y_eg: ...
        type_: Calculation type: "price", "delta" or "gamma".
            Default is "price".

    Returns:
        Zero-coupon bond price/delta/gamma.
    """
    if event_idx > maturity_idx:
        raise ValueError("event_idx > maturity_idx")
    # P(0,t): Zero-coupon bond price at time zero with maturity t.
    price1 = discount_curve[event_idx]
    # P(0,T): Zero-coupon bond price at time zero with maturity T.
    price2 = discount_curve[maturity_idx]
    # G(t,T): G-function.
    delta_t = event_grid[maturity_idx] - event_grid[event_idx]
    g = (1 - math.exp(-kappa * delta_t)) / kappa
    # y(t): y-function.
    y = y_eg[event_idx]
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    price = price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1
    if type_ == "price":
        return price
    elif type_ == "delta":
        return -g * price
    elif type_ == "gamma":
        return g ** 2 * price
    else:
        raise ValueError(f"Calculation type is unknown: {type_}")


def zcbond_general(spot: typing.Union[float, np.ndarray],
                   discount_curve: np.ndarray,
                   event_idx: int,
                   maturity_idx: int,
                   int_event_idx: np.ndarray,
                   int_grid: np.ndarray,
                   int_kappa_step: np.ndarray,
                   y_eg: np.ndarray,
                   type_: str = "price") -> typing.Union[float, np.ndarray]:
    """Calculate zero-coupon bond price, delta or gamma.

    Calculate price/delta/gamma of zero-coupon bond based at current
    time (event_idx) for maturity at time T (maturity_idx). See
    proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    Args:
        spot: Current pseudo short rate.
        discount_curve: Discount curve represented on event grid.
        event_idx: Event grid index corresponding to current time.
        maturity_idx: Event grid index corresponding to maturity.
        int_event_idx: ...
        int_grid: ...
        int_kappa_step: ...
        y_eg: ...
        type_: Calculation type: "price", "delta" or "gamma".
            Default is "price".

    Returns:
        Zero-coupon bond price/delta/gamma.
    """
    if event_idx > maturity_idx:
        raise ValueError("event_idx > maturity_idx")
    # P(0,t): Zero-coupon bond price at time zero with maturity t.
    price1 = discount_curve[event_idx]
    # P(0,T): Zero-coupon bond price at time zero with maturity T.
    price2 = discount_curve[maturity_idx]
    # Integration indices of the two relevant events.
    int_idx1 = int_event_idx[event_idx]
    int_idx2 = int_event_idx[maturity_idx] + 1
    # Slice of integration grid.
    int_grid = int_grid[int_idx1:int_idx2]
    # Slice of time-integrated kappa for each integration step.
    int_kappa = int_kappa_step[int_idx1:int_idx2]
    # G(t,T): G-function.
    integrand = np.exp(-np.cumsum(int_kappa))
    g = np.sum(misc.trapz(int_grid, integrand))
    # y(t): y-function.
    y = y_eg[event_idx]
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    price = price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1
    if type_ == "price":
        return price
    elif type_ == "delta":
        return -g * price
    elif type_ == "gamma":
        return g ** 2 * price
    else:
        raise ValueError(f"Calculation type is unknown: {type_}")
