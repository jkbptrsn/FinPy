import math
import typing

import numpy as np
from scipy.interpolate import UnivariateSpline

from models.hull_white import misc as misc_hw

from utils import global_types
from utils import misc


class SDEBasic:
    """Basic SDE class for 1-factor Hull-White model.

    See L.B.G. Andersen & V.V. Piterbarg 2010, chapter 10.1.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility strip.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise constant.
            "general": General time dependence.
            Default is "constant".
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 time_dependence: str = "constant",
                 int_step_size: float = 1 / 365):
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_step_size = int_step_size

        # Arrays used for exact discretization.
        self.rate_mean = np.zeros((event_grid.size, 2))
        self.rate_variance = np.zeros(event_grid.size)
        self.discount_mean = np.zeros((event_grid.size, 2))
        self.discount_variance = np.zeros(event_grid.size)
        self.covariance = np.zeros(event_grid.size)

        # Speed of mean reversion on event grid.
        self.kappa_eg = None
        # Volatility on event grid.
        self.vol_eg = None
        # Discount curve on event grid.
        self.discount_curve_eg = None

        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = None

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

    def initialization(self):
        """Initialization of Monte-Carlo engine.

        Calculate time-dependent mean and variance of the pseudo short
        rate and pseudo discount processes, respectively.
        """
        self._setup_int_grid()
        self._setup_model_parameters()
        self._calc_rate_mean()
        self._calc_rate_variance()
        self._calc_discount_mean()
        self._calc_discount_variance()
        self._calc_covariance()

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

        # Instantaneous forward rate on event grid.
        log_discount = np.log(self.discount_curve_eg)
        smoothing = 0
        log_discount_spline = \
            UnivariateSpline(self.event_grid, log_discount, s=smoothing)
        forward_rate = log_discount_spline.derivative()
        self.forward_rate_eg = -forward_rate(self.event_grid)

        # Kappa and vol are constant.
        if self.time_dependence == "constant":
            # y-function on event grid.
            self.y_eg = misc_hw.y_constant(self.kappa_eg[0],
                                           self.vol_eg[0],
                                           self.event_grid)
        # Kappa is constant and vol is piecewise constant.
        elif self.time_dependence == "piecewise":
            # y-function on event grid.
            self.y_eg = misc_hw.y_piecewise(self.kappa_eg[0],
                                            self.vol_eg,
                                            self.event_grid)
        # Kappa and vol have general time-dependence.
        elif self.time_dependence == "general":
            # Speed of mean reversion interpolated on integration grid.
            self.kappa_ig = self.kappa.interpolation(self.int_grid)
            # Volatility interpolated on integration grid.
            self.vol_ig = self.vol.interpolation(self.int_grid)
            # Integration of speed of mean reversion using trapezoidal rule.
            self.int_kappa_step = \
                np.append(0, misc.trapz(self.int_grid, self.kappa_ig))
            # y-function on event and integration grid.
            self.y_eg, self.y_ig = misc_hw.y_general(self.int_grid,
                                                     self.int_event_idx,
                                                     self.int_kappa_step,
                                                     self.vol_ig,
                                                     self.event_grid)
        else:
            raise ValueError(f"Time dependence unknown: "
                             f"{self.time_dependence}")

    def _calc_rate_mean(self):
        """Conditional mean of pseudo short rate process."""
        pass

    def _calc_rate_variance(self):
        """Conditional variance of pseudo short rate process."""
        pass

    def _rate_increment(self,
                        spot: typing.Union[float, np.ndarray],
                        event_idx: int,
                        normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment pseudo short rate process.

        The spot value is subtracted to get the increment.

        Args:
            spot: Pseudo short rate at event event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Incremented pseudo short rate process.
        """
        mean = \
            self.rate_mean[event_idx][0] * spot + self.rate_mean[event_idx][1]
        variance = self.rate_variance[event_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def _calc_discount_mean(self):
        """Conditional mean of pseudo discount process."""
        pass

    def _calc_discount_variance(self):
        """Conditional variance of pseudo discount process."""
        pass

    def _discount_increment(self,
                            rate_spot: typing.Union[float, np.ndarray],
                            event_idx: int,
                            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.

        Args:
            rate_spot: Pseudo short rate at event event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Incremented pseudo discount process.
        """
        mean = - rate_spot * self.discount_mean[event_idx][0] \
            - self.discount_mean[event_idx][1]
        variance = self.discount_variance[event_idx]
        return mean + math.sqrt(variance) * normal_rand

    def _calc_covariance(self):
        """Covariance between short rate and discount processes."""
        pass

    def _correlation(self,
                     event_idx: int) -> float:
        """Correlation between short rate and discount processes.

        Args:
            event_idx: Index on event grid.

        Returns:
            Correlation.
        """
        covariance = self.covariance[event_idx]
        rate_var = self.rate_variance[event_idx]
        discount_var = self.discount_variance[event_idx]
        return covariance / math.sqrt(rate_var * discount_var)

    def paths(self,
              spot: float,
              n_paths: int,
              rng: np.random.Generator = None,
              seed: int = None,
              antithetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo paths using exact discretization.

        Args:
            spot: Pseudo short rate at first event date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of pseudo short rate and pseudo discount
            processes represented on event_grid.
        """
        rate = np.zeros((self.event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self.event_grid.size, n_paths))
        if rng is None:
            rng = np.random.default_rng(seed)
        for event_idx in range(1, self.event_grid.size):
            correlation = self._correlation(event_idx)
            # Realizations of correlated normal random variables.
            rand_rate, rand_discount = \
                misc.cholesky_2d(correlation, n_paths, rng, antithetic)
            # Increment pseudo short rate process.
            rate[event_idx] = rate[event_idx - 1] \
                + self._rate_increment(rate[event_idx - 1], event_idx,
                                       rand_rate)
            # Increment pseudo discount process.
            discount[event_idx] = discount[event_idx - 1] \
                + self._discount_increment(rate[event_idx - 1], event_idx,
                                           rand_discount)
        # Get pseudo discount factors on event_grid.
        discount = np.exp(discount)
        return rate, discount


class SDEConstant(SDEBasic):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is given by
        dx_t = (y_t - kappa * x_t) * dt + vol * dW_t,
    where
        x_t = r_t - f(0,t) (f is the instantaneous forward rate).

    The speed of mean reversion and the volatility strip are constant.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility strip.
        discount_curve: Discount curve represented on event grid.
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
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, discount_curve, event_grid,
                         "constant", int_step_size)

        self.initialization()

    def _calc_rate_mean(self):
        """Conditional mean of pseudo short rate process.

        See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[0]
        # Values at initial event.
        self.rate_mean[0] = np.array([1, 0])
        # First term in Eq. (10.40).
        self.rate_mean[1:, 0] = np.exp(-kappa * np.diff(self.event_grid))
        # Second term in Eq. (10.40).
        self.rate_mean[:, 1] = \
            misc_hw.int_y_constant(kappa, vol, self.event_grid)

    def _calc_rate_variance(self):
        """Conditional variance of pseudo short rate process.

        See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[0]
        self.rate_variance[1:] = \
            vol ** 2 * (1 - np.exp(-2 * kappa * np.diff(self.event_grid))) \
            / (2 * kappa)

    def _calc_discount_mean(self):
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[0]
        # First term in Eq. (10.42).
        self.discount_mean[1:, 0] = \
            (1 - np.exp(-kappa * np.diff(self.event_grid))) / kappa
        # Second term in Eq. (10.42).
        self.discount_mean[:, 1] = \
            misc_hw.double_int_y_constant(kappa, vol, self.event_grid)

    def _calc_discount_variance(self):
        """Conditional variance of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).
        """
        self.discount_variance[1:] = 2 * self.discount_mean[1:, 1] \
            - self.y_eg[:-1] * self.discount_mean[1:, 0] ** 2

    def _calc_covariance(self):
        """Covariance between short rate and discount processes.

        See L.B.G. Andersen & V.V. Piterbarg 2010, lemma 10.1.11.
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[0]
        exp_kappa = np.exp(-kappa * np.diff(self.event_grid))
        self.covariance[1:] = \
            -vol ** 2 * (1 - exp_kappa) ** 2 / (2 * kappa ** 2)


class SDEPiecewise(SDEBasic):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is given by
        dx_t = (y_t - kappa * x_t) * dt + vol_t * dW_t,
    where
        x_t = r_t - f(0,t) (f is the instantaneous forward rate).

    The speed of mean reversion is constant and the volatility strip is
    piecewise constant.

    TODO: Implicit assumption that all vol-strip events are represented on the event grid.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility strip.
        discount_curve: Discount curve represented on event grid.
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
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, discount_curve, event_grid,
                         "constant", int_step_size)

        self.initialization()

    def _calc_rate_mean(self):
        """Conditional mean of pseudo short rate process.

        See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).
        """

        # TODO: double_y_piecewise, int_y_general, double_int_y_general

        kappa = self.kappa_eg[0]
        vol = self.vol_eg[0]
        # Values at initial event.
        self.rate_mean[0] = np.array([1, 0])
        # First term in Eq. (10.40).
        self.rate_mean[1:, 0] = np.exp(-kappa * np.diff(self.event_grid))
        # Second term in Eq. (10.40).
        exp_kappa_1 = np.exp(-2 * kappa * self.event_grid[1:])
        exp_kappa_2 = np.exp(-kappa * np.diff(self.event_grid))
        event_grid_sum = self.event_grid[1:] + self.event_grid[:-1]
        exp_kappa_3 = np.exp(-kappa * event_grid_sum)
        self.rate_mean[1:, 1] = \
            vol ** 2 * (1 + exp_kappa_1 - exp_kappa_2 - exp_kappa_3) \
            / (2 * kappa ** 2)

    def _calc_rate_variance(self):
        """Conditional variance of pseudo short rate process.

        See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[:-1]
        self.rate_variance[1:] = \
            vol ** 2 * (1 - np.exp(-2 * kappa * np.diff(self.event_grid))) \
            / (2 * kappa)

    def _calc_discount_mean(self):
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[0]
        # First term in Eq. (10.42).
        self.discount_mean[1:, 0] = \
            (1 - np.exp(-kappa * np.diff(self.event_grid))) / kappa
        # First term in Eq. (10.42).
        exp_kappa_1 = \
            (np.exp(-2 * kappa * self.event_grid[:-1])
             - np.exp(-2 * kappa * self.event_grid[1:])) / 2
        exp_kappa_2 = np.exp(-kappa * np.diff(self.event_grid)) - 1
        event_grid_sum = self.event_grid[1:] + self.event_grid[:-1]
        exp_kappa_3 = \
            np.exp(-kappa * event_grid_sum) \
            - np.exp(-2 * kappa * self.event_grid[:-1])
        self.discount_mean[1:, 1] = \
            vol ** 2 * (kappa * np.diff(self.event_grid) + exp_kappa_1
                        + exp_kappa_2 + exp_kappa_3) / (2 * kappa ** 3)

    def _calc_discount_variance(self):
        """Conditional variance of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).
        """
        self.discount_variance[1:] = 2 * self.discount_mean[1:, 1] \
            - self.y_eg[:-1] * self.discount_mean[1:, 0] ** 2

    def _calc_covariance(self):
        """Covariance between short rate and discount processes.

        See L.B.G. Andersen & V.V. Piterbarg 2010, lemma 10.1.11.
        """
        kappa = self.kappa_eg[0]
        vol = self.vol_eg[:-1]
        exp_kappa = np.exp(-kappa * np.diff(self.event_grid))
        self.covariance[1:] = \
            -vol ** 2 * (1 - exp_kappa) ** 2 / (2 * kappa ** 2)
