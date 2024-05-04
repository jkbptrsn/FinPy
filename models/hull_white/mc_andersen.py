import math
import typing

import numpy as np

from models.hull_white import misc as misc_hw
from utils import data_types
from utils import global_types
from utils import misc


def rate_adjustment(
        rate_paths: np.ndarray,
        adjustment: np.ndarray) -> np.ndarray:
    """Adjust pseudo rate paths.

    Assume that pseudo rate paths and instantaneous forward rate curve
    are represented on identical event grids.

    Args:
        rate_paths: Pseudo short rate along Monte-Carlo paths.
        adjustment: Instantaneous forward rate on event grid.

    Returns:
        Actual short rate paths.
    """
    return (rate_paths.transpose() + adjustment).transpose()


def discount_adjustment(
        discount_paths: np.ndarray,
        adjustment: np.ndarray) -> np.ndarray:
    """Adjust pseudo discount paths.

    Assume that pseudo discount paths and discount curve are represented
    on identical event grids.

    Args:
        discount_paths: Pseudo discount factor along Monte-Carlo paths.
        adjustment: Discount curve on event grid.

    Returns:
        Actual discount paths.
    """
    tmp = discount_paths.transpose() * adjustment
    return tmp.transpose()


class SdeExact:
    """SDE for pseudo short rate process in 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = (y_t - kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa_t is the speed of mean reversion and vol_t denotes the
    volatility. W_t is a Brownian motion process under the risk-neutral
    measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t),
    where f is the instantaneous forward rate.

    See Andersen & Piterbarg (2010), Section 10.1.

    Monte-Carlo paths constructed using exact discretization.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            event_grid: np.ndarray,
            time_dependence: str = "piecewise",
            int_dt: float = 1 / 52):
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
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

        # Arrays used for exact discretization.
        self.rate_mean = np.zeros((event_grid.size, 2))
        self.rate_variance = np.zeros(event_grid.size)
        self.discount_mean = np.zeros((event_grid.size, 2))
        self.discount_variance = np.zeros(event_grid.size)
        self.covariance = np.zeros(event_grid.size)

        self.rate_paths = None
        self.discount_paths = None
        self.mc_estimate = None
        self.mc_error = None

    def initialization(self) -> None:
        """Initialization of Monte-Carlo engine.

        Calculate time-dependent mean and variance of the pseudo short
        rate and pseudo discount processes, respectively.
        """
        if self.time_dependence == "general":
            self._setup_int_grid()
        self._setup_model_parameters()
        self._calc_rate_mean()
        self._calc_rate_variance()
        self._calc_discount_mean()
        self._calc_discount_variance()
        self._calc_covariance()

    def _setup_int_grid(self) -> None:
        """Set up time grid for numerical integration."""
        self.int_grid, self.int_event_idx = \
            misc_hw.integration_grid(self.event_grid, self.int_dt)

    def _setup_model_parameters(self) -> None:
        """Set up model parameters on event and integration grids."""
        misc_hw.setup_model_parameters(self)

    def _calc_rate_mean(self) -> None:
        """Conditional mean of pseudo short rate process."""
        pass

    def _calc_rate_variance(self) -> None:
        """Conditional variance of pseudo short rate process."""
        pass

    def _rate_increment(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment pseudo short rate process.

        The spot value is subtracted to get the increment.

        Args:
            spot: Pseudo short rate at event event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of standard normal random
                variables.

        Returns:
            Increment of pseudo short rate process.
        """
        mean = \
            self.rate_mean[event_idx, 0] * spot + self.rate_mean[event_idx, 1]
        variance = self.rate_variance[event_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def _calc_discount_mean(self) -> None:
        """Conditional mean of pseudo discount process."""
        pass

    def _calc_discount_variance(self) -> None:
        """Conditional variance of pseudo discount process."""
        pass

    def _discount_increment(
            self,
            rate_spot: typing.Union[float, np.ndarray],
            event_idx: int,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.

        Args:
            rate_spot: Pseudo short rate at event event_idx - 1.
            event_idx: Index on event grid.
            normal_rand: Realizations of standard normal random
                variables.

        Returns:
            Increment of pseudo discount process.
        """
        mean = - rate_spot * self.discount_mean[event_idx, 0] \
            - self.discount_mean[event_idx, 1]
        variance = self.discount_variance[event_idx]
        return mean + math.sqrt(variance) * normal_rand

    def _calc_covariance(self) -> None:
        """Covariance between short rate and discount processes."""
        pass

    def _correlation(
            self,
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

    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Generation of Monte-Carlo paths using exact discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of correlated pseudo short rate and pseudo
            discount processes represented on event_grid.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        # Paths of rate process.
        r_paths = np.zeros((self.event_grid.size, n_paths))
        r_paths[0] = spot
        # Paths of discount process.
        d_paths = np.zeros((self.event_grid.size, n_paths))
        for event_idx in range(1, self.event_grid.size):
            correlation = self._correlation(event_idx)
            # Realizations of standard normal random variables.
            x_rate, x_discount = misc.cholesky_2d(
                correlation, n_paths, rng, antithetic)
            # Increment pseudo short rate process, and update.
            r_increment = self._rate_increment(
                r_paths[event_idx - 1], event_idx, x_rate)
            r_paths[event_idx] = r_paths[event_idx - 1] + r_increment
            # Increment pseudo discount process, and update.
            d_increment = self._discount_increment(
                r_paths[event_idx - 1], event_idx, x_discount)
            d_paths[event_idx] = d_paths[event_idx - 1] + d_increment
        # Get pseudo discount factors on event_grid.
        d_paths = np.exp(d_paths)
        # Update.
        self.rate_paths = r_paths
        self.discount_paths = d_paths

    @staticmethod
    def rate_adjustment(
            rate_paths: np.ndarray,
            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo rate paths."""
        return rate_adjustment(rate_paths, adjustment)

    @staticmethod
    def discount_adjustment(
            discount_paths: np.ndarray,
            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo discount paths."""
        return discount_adjustment(discount_paths, adjustment)


class SdeExactConstant(SdeExact):
    """SDE class for 1-factor Hull-White model.

    Monte-Carlo paths constructed using exact discretization.

    The speed of mean reversion and volatility is constant.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        event_grid: Event dates as year fractions from as-of date.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            event_grid: np.ndarray):
        super().__init__(kappa, vol, discount_curve, event_grid, "constant")

        self.initialization()

    def _calc_rate_mean(self) -> None:
        """Conditional mean of pseudo short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.40).
        """
        # First term in Eq. (10.40).
        self.rate_mean[0, 0] = 1
        self.rate_mean[1:, 0] = np.exp(-np.diff(self.int_kappa_eg))
        # Second term in Eq. (10.40).
        self.rate_mean[:, 1] = misc_hw.int_y_constant(
            self.kappa_eg[0], self.vol_eg[0], self.event_grid)

    def _calc_rate_variance(self) -> None:
        """Conditional variance of pseudo short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.41).
        """
        two_kappa = 2.0 * self.kappa_eg[0]
        exp_kappa = np.exp(-two_kappa * np.diff(self.event_grid))
        self.rate_variance[0] = 0
        self.rate_variance[1:] = \
            self.vol_eg[0] ** 2 * (1 - exp_kappa) / two_kappa

    def _calc_discount_mean(self) -> None:
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.
        See Andersen & Piterbarg (2010), Eq. (10.42).
        """
        # First term in Eq. (10.42).
        self.discount_mean[0, :] = 0
        self.discount_mean[1:, 0] = \
            (self.g_eg[1:] - self.g_eg[:-1]) * np.exp(self.int_kappa_eg[:-1])
        # Second term in Eq. (10.42).
        self.discount_mean[:, 1] = misc_hw.int_int_y_constant(
            self.kappa_eg[0], self.vol_eg[0], self.event_grid)

    def _calc_discount_variance(self) -> None:
        """Conditional variance of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.
        See Andersen & Piterbarg (2010), Eq. (10.43).
        """
        self.discount_variance[0] = 0
        self.discount_variance[1:] = 2 * self.discount_mean[1:, 1] \
            - self.y_eg[:-1] * self.discount_mean[1:, 0] ** 2

    def _calc_covariance(self) -> None:
        """Covariance between short rate and discount processes.

        See Andersen & Piterbarg (2010), Lemma 10.1.11.
        """
        exp_kappa_sq = \
            (1 - np.exp(-self.kappa_eg[0] * np.diff(self.event_grid))) ** 2
        self.covariance[0] = 0
        self.covariance[1:] = \
            -self.vol_eg[0] ** 2 * exp_kappa_sq / (2 * self.kappa_eg[0] ** 2)


class SdeExactPiecewise(SdeExact):
    """SDE class for 1-factor Hull-White model.

    Monte-Carlo paths constructed using exact discretization.

    The speed of mean reversion is constant and the volatility is
    piecewise constant.

    Note: Implicit assumption that all vol-strip events are represented
    on event grid.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        event_grid: Event dates as year fractions from as-of date.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            event_grid: np.ndarray):
        super().__init__(kappa, vol, discount_curve, event_grid, "piecewise")

        self.initialization()

    def _calc_rate_mean(self) -> None:
        """Conditional mean of pseudo short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.40).
        """
        # First term in Eq. (10.40).
        self.rate_mean[0, 0] = 1
        self.rate_mean[1:, 0] = np.exp(-np.diff(self.int_kappa_eg))
        # Second term in Eq. (10.40).
        self.rate_mean[:, 1] = misc_hw.int_y_piecewise(
            self.kappa_eg[0], self.vol_eg, self.event_grid)

    def _calc_rate_variance(self) -> None:
        """Conditional variance of pseudo short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.41).
        """
        two_kappa = 2.0 * self.kappa_eg[0]
        exp_kappa = np.exp(-two_kappa * np.diff(self.event_grid))
        self.rate_variance[0] = 0
        self.rate_variance[1:] = \
            self.vol_eg[:-1] ** 2 * (1 - exp_kappa) / two_kappa

    def _calc_discount_mean(self) -> None:
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        Andersen & Piterbarg (2010), Eq. (10.42).
        """
        # First term in Eq. (10.42).
        self.discount_mean[0, :] = 0
        self.discount_mean[1:, 0] = \
            (self.g_eg[1:] - self.g_eg[:-1]) * np.exp(self.int_kappa_eg[:-1])
        # Second term in Eq. (10.42).
        self.discount_mean[:, 1] = misc_hw.int_int_y_piecewise(
            self.kappa_eg[0], self.vol_eg, self.event_grid)

    def _calc_discount_variance(self) -> None:
        """Conditional variance of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        Andersen & Piterbarg (2010), Eq. (10.43).
        """
        self.discount_variance[0] = 0
        self.discount_variance[1:] = 2 * self.discount_mean[1:, 1] \
            - self.y_eg[:-1] * self.discount_mean[1:, 0] ** 2

    def _calc_covariance(self) -> None:
        """Covariance between short rate and discount processes.

        See Andersen & Piterbarg (2010), Lemma 10.1.11.
        """
        exp_kappa_sq = \
            (1 - np.exp(-self.kappa_eg[0] * np.diff(self.event_grid))) ** 2
        self.covariance[0] = 0
        self.covariance[1:] = \
            -self.vol_eg[:-1] ** 2 * exp_kappa_sq / (2 * self.kappa_eg[0] ** 2)


class SdeExactGeneral(SdeExact):
    """SDE class for 1-factor Hull-White model.

    Monte-Carlo paths constructed using exact discretization.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    Note: Implicit assumption that all vol-strip events are represented
    on event grid.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        event_grid: Event dates as year fractions from as-of date.
        int_dt: Integration step size. Default is 1 / 52.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            event_grid: np.ndarray,
            int_dt: float = 1 / 52):
        super().__init__(
            kappa, vol, discount_curve, event_grid, "general", int_dt)

        self.initialization()

    def _calc_rate_mean(self) -> None:
        """Conditional mean of pseudo short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.40).
        """
        # First term in Eq. (10.40).
        self.rate_mean[0, 0] = 1
        self.rate_mean[1:, 0] = np.exp(-np.diff(self.int_kappa_eg))
        # Second term in Eq. (10.40).
        self.rate_mean[:, 1] = misc_hw.int_y_general(
            self.int_grid, self.int_event_idx, self.int_kappa_step_ig,
            self.vol_ig, self.event_grid)

    def _calc_rate_variance(self) -> None:
        """Conditional variance of pseudo short rate process.

        See Andersen & Piterbarg (2010), Eq. (10.41).
        """
        self.rate_variance[0] = 0
        for event_idx in range(1, self.event_grid.size):
            # Integration indices of two adjacent events.
            idx1 = self.int_event_idx[event_idx - 1]
            idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid.
            int_grid_tmp = self.int_grid[idx1:idx2]

            # TODO: Remove!
            # Slice of time-integrated kappa for each integration step.
#            int_kappa = np.append(self.int_kappa_step_ig[idx1 + 1:idx2], 0)

            int_kappa = np.append(0, self.int_kappa_step_ig[idx1 + 1:idx2])
            # Cumulative sum from "right to left".
            int_kappa = np.flip(np.cumsum(np.flip(int_kappa)))
            # Shift to the left.
            int_kappa[:-1] = int_kappa[1:]
            int_kappa[-1] = 0
            integrand = (np.exp(-int_kappa) * self.vol_ig[idx1:idx2]) ** 2
            variance = np.sum(misc.trapz(int_grid_tmp, integrand))
            self.rate_variance[event_idx] = variance

    def _calc_discount_mean(self) -> None:
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.
        See Andersen & Piterbarg (2010), Eq. (10.42).
        """
        # First term in Eq. (10.42).
        self.discount_mean[0, :] = 0
        self.discount_mean[1:, 0] = \
            (self.g_eg[1:] - self.g_eg[:-1]) * np.exp(self.int_kappa_eg[:-1])
        # Second term in Eq. (10.42).
        self.discount_mean[:, 1] = \
            misc_hw.int_int_y_general(
                self.int_grid, self.int_event_idx, self.int_kappa_step_ig,
                self.vol_ig, self.event_grid)

    def _calc_discount_variance(self) -> None:
        """Conditional variance of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.
        See Andersen & Piterbarg (2010), Eq. (10.43).
        """
        self.discount_variance[0] = 0
        self.discount_variance[1:] = 2 * self.discount_mean[1:, 1] \
            - self.y_eg[:-1] * self.discount_mean[1:, 0] ** 2

    def _calc_covariance(self) -> None:
        """Covariance between short rate and discount processes.

        See Andersen & Piterbarg (2010), Lemma 10.1.11.
        """
        self.covariance[0] = 0
        for event_idx in range(1, self.event_grid.size):
            # Integration indices of two adjacent events.
            idx1 = self.int_event_idx[event_idx - 1]
            idx2 = self.int_event_idx[event_idx] + 1
            # Slice of time-integrated kappa for each integration step.
            int_kappa = np.append(0, self.int_kappa_step_ig[idx1 + 1:idx2])
            # Cumulative sum from "right to left".
            int_kappa = np.flip(np.cumsum(np.flip(int_kappa)))
            # Shift to the left.
            int_kappa[:-1] = int_kappa[1:]
            int_kappa[-1] = 0
            exp_kappa = np.exp(-int_kappa)
            cov = np.array(0)
            for idx in range(idx1 + 1, idx2):
                int_grid_tmp = self.int_grid[idx1:idx + 1]
                int_kappa_tmp = (
                    np.append(0, self.int_kappa_step_ig[idx1 + 1:idx + 1]))
                # Cumulative sum from "right to left".
                int_kappa_tmp = np.flip(np.cumsum(np.flip(int_kappa_tmp)))
                # Shift to the left.
                int_kappa_tmp[:-1] = int_kappa_tmp[1:]
                int_kappa_tmp[-1] = 0
                integrand = self.vol_ig[idx1:idx + 1] ** 2 * \
                    np.exp(-int_kappa_tmp) * exp_kappa[:idx + 1 - idx1]
                cov = np.append(cov,
                                np.sum(misc.trapz(int_grid_tmp, integrand)))
            # Slice of integration grid.
            int_grid = self.int_grid[idx1:idx2]
            self.covariance[event_idx] = -np.sum(misc.trapz(int_grid, cov))


class SdeEuler:
    """SDE for pseudo short rate process in 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = (y_t - kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa_t is the speed of mean reversion and vol_t denotes the
    volatility. W_t is a Brownian motion process under the risk-neutral
    measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t),
    where f is the instantaneous forward rate.

    See Andersen & Piterbarg (2010), Section 10.1.

    Monte-Carlo paths constructed using Euler-Maruyama discretization.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            event_grid: np.ndarray,
            time_dependence: str = "piecewise",
            int_dt: float = 1 / 52):
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
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

        self.initialization()

        self.rate_paths = None
        self.discount_paths = None
        self.mc_estimate = None
        self.mc_error = None

    def initialization(self) -> None:
        """Initialization of Monte-Carlo engine.

        Calculate time-dependent mean and variance of the pseudo short
        rate and pseudo discount processes, respectively.
        """
        if self.time_dependence == "general":
            self._setup_int_grid()
        self._setup_model_parameters()

    def _setup_int_grid(self) -> None:
        """Set up time grid for numerical integration."""
        self.int_grid, self.int_event_idx = \
            misc_hw.integration_grid(self.event_grid, self.int_dt)

    def _setup_model_parameters(self) -> None:
        """Set up model parameters on event and integration grids."""
        misc_hw.setup_model_parameters(self)

    def _rate_increment(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            dt: float,
            normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment pseudo short rate process.

        See Andersen & Piterbarg (2010), Section 10.1.6.2.

        The spot value is subtracted to get the increment.

        Args:
            spot: Pseudo short rate at event event_idx.
            event_idx: Index on event grid.
            normal_rand: Realizations of standard normal random
                variables.

        Returns:
            Increment of pseudo short rate process.
        """
        kappa = self.kappa_eg[event_idx]
        exp_kappa = math.exp(-kappa * dt)
        wiener_increment = math.sqrt(dt) * normal_rand
        rate_increment = exp_kappa * spot \
            + (1 - exp_kappa) * self.y_eg[event_idx] / kappa \
            + self.vol_eg[event_idx] * wiener_increment - spot
        return rate_increment

    def paths(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Generation of Monte-Carlo paths using Euler discretization.

        Args:
            spot: Pseudo short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of correlated pseudo short rate and pseudo
            discount processes represented on event_grid.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        # Paths of rate process.
        r_paths = np.zeros((self.event_grid.size, n_paths))
        r_paths[0] = spot
        # Paths of discount process.
        d_paths = np.zeros((self.event_grid.size, n_paths))
        for idx, dt in enumerate(np.diff(self.event_grid)):
            event_idx = idx + 1
            # Realizations of standard normal random variables.
            x_rate = misc.normal_realizations(n_paths, rng, antithetic)
            # Increment of rate process, and update.
            r_increment = self._rate_increment(
                r_paths[event_idx - 1], event_idx - 1, dt, x_rate)
            r_paths[event_idx] = r_paths[event_idx - 1] + r_increment
            # Increment of discount process, and update.
            d_increment = -r_paths[event_idx - 1] * dt
            d_paths[event_idx] = d_paths[event_idx - 1] + d_increment
        # Get actual discount factors on event_grid.
        d_paths = np.exp(d_paths)
        # Update.
        self.rate_paths = r_paths
        self.discount_paths = d_paths

    @staticmethod
    def rate_adjustment(
            rate_paths: np.ndarray,
            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo rate paths."""
        return rate_adjustment(rate_paths, adjustment)

    @staticmethod
    def discount_adjustment(
            discount_paths: np.ndarray,
            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo discount paths."""
        return discount_adjustment(discount_paths, adjustment)
