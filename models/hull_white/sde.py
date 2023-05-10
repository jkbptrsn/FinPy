import math
import numpy as np

from models import sde
from utils import global_types
from utils import misc

# kappa_int_grid and vol_int_grid are not used.
# Should be used in an Euler propagation...

###############################################################################


def rate_adjustment(event_grid: np.ndarray,
                    rate_paths: np.ndarray,
                    forward_curve: misc.DiscreteFunc,
                    replace: bool = False) -> (None, np.ndarray):
    """Adjust pseudo rate path for each Monte-Carlo scenario. Assume
    that pseudo rate paths are represented on event_grid."""
    forward_curve_grid = forward_curve.interpolation(event_grid)
    if replace:
        for event_idx in range(event_grid.size):
            rate_paths[event_idx, :] += forward_curve_grid[event_idx]
    else:
        rate_paths_adj = np.zeros(rate_paths.shape)
        for event_idx in range(event_grid.size):
            rate_paths_adj[event_idx, :] = \
                rate_paths[event_idx, :] + forward_curve_grid[event_idx]
        return rate_paths_adj


###############################################################################


def discount_adjustment(discount_paths: np.ndarray,
                        discount_curve: misc.DiscreteFunc) -> np.ndarray:
    """Adjust pseudo discount paths.

    Assume that discount curve and pseudo discount paths are represented
    on event_grid.

    Args:
        discount_paths:
        discount_curve:

    Returns:

    """
    adjustment = discount_paths.transpose() * discount_curve.values
    return adjustment.transpose()


class SDEBasic(sde.SDE):
    """Basic SDE class for the 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        self.kappa = kappa
        self.vol = vol
        self.event_grid = event_grid
        self.int_step_size = int_step_size

        self.model_name = global_types.Model.HULL_WHITE_1F

        # Arrays used for exact discretization.
        self.rate_mean = np.zeros((event_grid.size, 2))
        self.rate_variance = np.zeros(event_grid.size)
        self.discount_mean = np.zeros((event_grid.size, 2))
        self.discount_variance = np.zeros(event_grid.size)
        self.covariance = np.zeros(event_grid.size)

        # Integration grid.
        self.int_grid = None
        # Indices of event dates on integration grid.
        self.int_event_idx = None
        # y-function on integration and event grids. See Eq. (10.17),
        # L.B.G. Andersen & V.V. Piterbarg 2010.
        self.y_int_grid = None
        self.y_event_grid = np.zeros(event_grid.size)

    def __repr__(self):
        return f"{self.model_name} SDE object"

    def initialization(self):
        """Initialization of the Monte-Carlo engine.

        Calculate time-dependent mean and variance of the pseudo short
        rate and pseudo discount processes, respectively.
        """
        self._setup_int_grid()
        self._setup_kappa_vol_y()
        self._calc_rate_mean()
        self._calc_rate_variance()
        self._calc_discount_mean()
        self._calc_discount_variance()
        self._calc_covariance()

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

    def _setup_kappa_vol_y(self):
        """Set-up speed of mean reversion, volatility and y-function."""
        pass

    def _calc_rate_mean(self):
        """Conditional mean of pseudo short rate process."""
        pass

    def _calc_rate_variance(self):
        """Conditional variance of pseudo short rate process."""
        pass

    def _rate_increment(self,
                        spot: (float, np.ndarray),
                        time_idx: int,
                        normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo short rate process.

        The spot value is subtracted to get the increment.

        Args:
            spot: Pseudo short rate at time corresponding to time index.
            time_idx: Time index.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Incremented pseudo short rate process.
        """
        mean = self.rate_mean[time_idx][0] * spot + self.rate_mean[time_idx][1]
        variance = self.rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def _calc_discount_mean(self):
        """Conditional mean of pseudo discount process."""
        pass

    def _calc_discount_variance(self):
        """Conditional variance of pseudo discount process."""
        pass

    def _discount_increment(self,
                            rate_spot: (float, np.ndarray),
                            time_idx: int,
                            normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du.

        Args:
            rate_spot: Pseudo short rate at time corresponding to time
                index.
            time_idx: Time index.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Incremented pseudo discount process.
        """
        mean = \
            - rate_spot * self.discount_mean[time_idx][0] \
            - self.discount_mean[time_idx][1]
        variance = self.discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def _calc_covariance(self):
        """Covariance between short rate and discount processes."""
        pass

    def _correlation(self,
                     time_idx: int) -> float:
        """Correlation between pseudo short rate and discount processes.

        Args:
            time_idx: Time index.

        Returns:
            Correlation at time corresponding to time index.
        """
        covariance = self.covariance[time_idx]
        rate_var = self.rate_variance[time_idx]
        discount_var = self.discount_variance[time_idx]
        return covariance / math.sqrt(rate_var * discount_var)

    def paths(self,
              spot: float,
              n_paths: int,
              rng: np.random.Generator = None,
              seed: int = None,
              antithetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo paths using exact discretization.

        Args:
            spot: Pseudo short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of pseudo short rate and discount processes
            represented on event_grid.
        """
        rate = np.zeros((self.event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self.event_grid.size, n_paths))
        if rng is None:
            rng = np.random.default_rng(seed)
        for time_idx in range(1, self.event_grid.size):
            correlation = self._correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, rng, antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self._rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self._discount_increment(rate[time_idx - 1], time_idx,
                                           x_discount)
        # Get pseudo discount factors on event_grid.
        discount = np.exp(discount)
        return rate, discount

    def paths_sobol_test(self,
                         spot: float,
                         n_paths: int,
                         sobol_norm) -> tuple[np.ndarray, np.ndarray]:
        """Quasi Monte-Carlo paths using exact discretization.

        Sobol sequences are used in the path generation.
        """
        rate = np.zeros((self.event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self.event_grid.size, n_paths))
        for time_idx in range(1, self.event_grid.size):
            correlation = self._correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d_sobol_test(correlation, sobol_norm, time_idx)
            rate[time_idx] = rate[time_idx - 1] \
                + self._rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self._discount_increment(rate[time_idx - 1], time_idx,
                                           x_discount)
        # Get pseudo discount factors on event_grid.
        discount = np.exp(discount)
        return rate, discount


class SDEConstant(SDEBasic):
    """SDE for the pseudo short rate in the 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = (y_t - kappa * x_t) * dt + vol * dW_t,
    where
        x_t = r_t - f(0,t) (f is the instantaneous forward rate). See
    proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    Attributes:
        kappa: Speed of mean reversion -- assumed to be constant.
        vol: Volatility -- assumed to be constant.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)

        self.initialization()

    def _setup_kappa_vol_y(self):
        """Set-up speed of mean reversion, volatility and y-function.

        See proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion on integration grid. kappa is assumed
        # to be constant.
        kappa = self.kappa.values[0]
        self.kappa_int_grid = kappa + 0 * self.int_grid
        # Volatility on integration grid. vol is assumed to be constant.
        vol = self.vol.values[0]
        self.vol_int_grid = vol + 0 * self.int_grid
        # Calculation of y-function on integration grid.
        exp_kappa = np.exp(-2 * kappa * self.int_grid)
        self.y_int_grid = vol ** 2 * (1 - exp_kappa) / (2 * kappa)
        # Save y-function on event grid.
        for idx, event_idx in enumerate(self.int_event_idx):
            self.y_event_grid[idx] = self.y_int_grid[event_idx]

    def _calc_rate_mean(self):
        """Conditional mean of pseudo short rate process.

        See Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion is assumed to be constant.
        kappa = self.kappa.values[0]
        # Volatility is assumed to be constant.
        vol = self.vol.values[0]
        self.rate_mean[0] = np.array([1, 0])
        self.rate_mean[1:, 0] = np.exp(-kappa * np.diff(self.event_grid))
        exp_kappa_1 = np.exp(-2 * kappa * self.event_grid[1:])
        exp_kappa_2 = np.exp(-kappa * np.diff(self.event_grid))
        event_grid_sum = self.event_grid[1:] + self.event_grid[:-1]
        exp_kappa_3 = np.exp(-kappa * event_grid_sum)
        self.rate_mean[1:, 1] = \
            vol ** 2 * (1 + exp_kappa_1 - exp_kappa_2 - exp_kappa_3) \
            / (2 * kappa ** 2)

    def _calc_rate_variance(self):
        """Conditional variance of pseudo short rate process.

        See Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion is assumed to be constant.
        kappa = self.kappa.values[0]
        # Volatility is assumed to be constant.
        vol = self.vol.values[0]
        self.rate_variance[1:] = \
            vol ** 2 * (1 - np.exp(-2 * kappa * np.diff(self.event_grid))) \
            / (2 * kappa)

    def _calc_discount_mean(self):
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        Eq. (10.42), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion is assumed to be constant.
        kappa = self.kappa.values[0]
        # Volatility is assumed to be constant.
        vol = self.vol.values[0]
        self.discount_mean[1:, 0] = \
            (1 - np.exp(-kappa * np.diff(self.event_grid))) / kappa
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
        Eq. (10.43), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        self.discount_variance[1:] = \
            2 * self.discount_mean[1:, 1] \
            - self.y_event_grid[:-1] * self.discount_mean[1:, 0] ** 2

    def _calc_covariance(self):
        """Covariance between between short rate and discount processes.

        See lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion is assumed to be constant.
        kappa = self.kappa.values[0]
        # Volatility is assumed to be constant.
        vol = self.vol.values[0]
        exp_kappa_1 = np.exp(-2 * kappa * np.diff(self.event_grid))
        exp_kappa_2 = np.exp(-kappa * np.diff(self.event_grid))
        self.covariance[1:] = \
            -vol ** 2 * (1 + exp_kappa_1 - 2 * exp_kappa_2) / (2 * kappa ** 2)


class SDE(SDEBasic):
    """SDE for the pseudo short rate in the 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = (y_t - kappa_t * x_t) * dt + vol_t * dW_t,
    where
        x_t = r_t - f(0,t) (f is the instantaneous forward rate). See
    proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.

    Note: If kappa is >1, the integration step size should be decreased!
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)

        # Array only used in initialization of the SDE object.
        self.int_kappa_step = None
        self.initialization()
        # Not deleted due to use in ZCBond class!
        # del self.int_kappa_step

    def _setup_kappa_vol_y(self):
        """Set-up speed of mean reversion, volatility and y-function.

        See proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion interpolated on integration grid.
        self.kappa_int_grid = self.kappa.interpolation(self.int_grid)
        # Volatility interpolated on integration grid.
        self.vol_int_grid = self.vol.interpolation(self.int_grid)
        # Integration of speed of mean reversion using trapezoidal rule.
        self.int_kappa_step = \
            np.append(0, misc.trapz(self.int_grid, self.kappa_int_grid))
        # Calculation of y-function on integration grid.
        self.y_int_grid = np.zeros(self.int_grid.size)
        for idx in range(1, self.int_grid.size):
            # int_u^t_{idx} kappa_s ds.
            int_kappa = self.int_kappa_step[:idx + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            int_kappa[:-1] = int_kappa[1:]
            int_kappa[-1] = 0
            # Integrand in expression for y.
            integrand = \
                np.exp(-2 * int_kappa) * self.vol_int_grid[:idx + 1] ** 2
            self.y_int_grid[idx] = \
                np.sum(misc.trapz(self.int_grid[:idx + 1], integrand))
        # Save y-function on event grid.
        for idx, event_idx in enumerate(self.int_event_idx):
            self.y_event_grid[idx] = self.y_int_grid[event_idx]

    def _calc_rate_mean(self):
        """Conditional mean of pseudo short rate process.

        See Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        self.rate_mean[0] = [1, 0]
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent events.
            idx1 = self.int_event_idx[event_idx - 1]
            idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid.
            int_grid = self.int_grid[idx1:idx2]
            # Slice of time-integrated kappa for each integration step.
            int_kappa = np.append(self.int_kappa_step[idx1 + 1:idx2], 0)
            factor1 = math.exp(-np.sum(int_kappa))
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = np.exp(-int_kappa) * self.y_int_grid[idx1:idx2]
            factor2 = np.sum(misc.trapz(int_grid, integrand))
            self.rate_mean[event_idx] = [factor1, factor2]

    def _calc_rate_variance(self):
        """Conditional variance of pseudo short rate process.

        See Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent events.
            idx1 = self.int_event_idx[event_idx - 1]
            idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid.
            int_grid = self.int_grid[idx1:idx2]
            # Slice of time-integrated kappa for each integration step.
            int_kappa = np.append(self.int_kappa_step[idx1 + 1:idx2], 0)
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = np.exp(-int_kappa) * self.vol_int_grid[idx1:idx2]
            integrand = integrand ** 2
            variance = np.sum(misc.trapz(int_grid, integrand))
            self.rate_variance[event_idx] = variance

    def _calc_discount_mean(self):
        """Conditional mean of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        Eq. (10.42), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent events.
            idx1 = self.int_event_idx[event_idx - 1]
            idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid.
            int_grid = self.int_grid[idx1:idx2]
            # Slice of time-integrated kappa for each integration step.
            int_kappa = np.append(0, self.int_kappa_step[idx1 + 1:idx2])
            # G-function in Eq. (10.18).
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            term1 = np.sum(misc.trapz(int_grid, integrand))
            # Double time integral in Eq. (10.42).
            term2 = np.array(0)
            for idx in range(idx1 + 1, idx2):
                int_grid_tmp = self.int_grid[idx1:idx + 1]
                int_kappa_tmp = \
                    np.append(self.int_kappa_step[idx1 + 1:idx + 1], 0)
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self.y_int_grid[idx1:idx + 1]
                term2 = \
                    np.append(term2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            term2 = np.sum(misc.trapz(int_grid, term2))
            self.discount_mean[event_idx] = [term1, term2]

    def _calc_discount_variance(self):
        """Conditional variance of pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} x_u du. See
        Eq. (10.43), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Double time integral in Eq. (10.43).
            term1 = 2 * self.discount_mean[event_idx][1]
            # G-function in Eq. (10.18).
            term2 = self.discount_mean[event_idx][0]
            # Integration index
            idx = self.int_event_idx[event_idx - 1]
            # Second term in Eq. (10.43).
            term2 = self.y_int_grid[idx] * term2 ** 2
            self.discount_variance[event_idx] = term1 - term2

    def _calc_covariance(self):
        """Covariance between between short rate and discount processes.

        See lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent events.
            idx1 = self.int_event_idx[event_idx - 1]
            idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid.
            int_grid = self.int_grid[idx1:idx2]
            # Slice of time-integrated kappa for each integration step.
            int_kappa = np.append(0, self.int_kappa_step[idx1 + 1:idx2])
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            exp_kappa = np.exp(-int_kappa)
            cov = np.array(0)
            for idx in range(idx1 + 1, idx2):
                int_grid_tmp = self.int_grid[idx1:idx + 1]
                int_kappa_tmp = \
                    np.append(self.int_kappa_step[idx1 + 1:idx + 1], 0)
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) \
                    * self.vol_int_grid[idx1:idx + 1] ** 2 \
                    * exp_kappa[:idx + 1 - idx1]
                cov = np.append(cov,
                                np.sum(misc.trapz(int_grid_tmp, integrand)))
            self.covariance[event_idx] = -np.sum(misc.trapz(int_grid, cov))

    def calc_rate_mean_custom(self, event_idx_1, event_idx_2):
        """Conditional mean of pseudo short rate process.

        See Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Integration indices of the two events.
        idx1 = self.int_event_idx[event_idx_1]
        idx2 = self.int_event_idx[event_idx_2] + 1
        # Slice of integration grid.
        int_grid = self.int_grid[idx1:idx2]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = np.append(self.int_kappa_step[idx1 + 1:idx2], 0)
        factor1 = math.exp(-np.sum(int_kappa))
        int_kappa = np.cumsum(int_kappa[::-1])[::-1]
        integrand = np.exp(-int_kappa) * self.y_int_grid[idx1:idx2]
        factor2 = np.sum(misc.trapz(int_grid, integrand))
        return factor1, factor2

    def calc_rate_variance_custom(self, event_idx_1, event_idx_2):
        """Conditional variance of pseudo short rate process.

        See Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Integration indices of the two events.
        idx1 = self.int_event_idx[event_idx_1]
        idx2 = self.int_event_idx[event_idx_2] + 1
        # Slice of integration grid.
        int_grid = self.int_grid[idx1:idx2]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = np.append(self.int_kappa_step[idx1 + 1:idx2], 0)
        int_kappa = np.cumsum(int_kappa[::-1])[::-1]
        integrand = np.exp(-int_kappa) * self.vol_int_grid[idx1:idx2]
        integrand = integrand ** 2
        return np.sum(misc.trapz(int_grid, integrand))


class SDEPelsser(SDE):
    """SDE for the pseudo short rate in the 1-factor Hull-White model.

    The pseudo short rate is defined by
        dz_t = - kappa_t * z_t * dt + vol_t * dW_t,
    where
        z_t = r_t - alpha_t. See
    proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.

    Note: If kappa is >1, the integration step size should be decreased!
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)

        # Array only used in initialization of the SDE object.
        self.int_kappa_step = None
        self.initialization()
        # Not deleted due to use in ZCBond class!
        # del self.int_kappa_step

    def _rate_increment(self,
                        spot: (float, np.ndarray),
                        time_idx: int,
                        normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo short rate process.

        The spot value is subtracted to get the increment.

        Args:
            spot: Pseudo short rate at time corresponding to time index.
            time_idx: Time index.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Incremented pseudo short rate process.
        """
        mean = self.rate_mean[time_idx][0] * spot
        variance = self.rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def _discount_increment(self,
                            rate_spot: (float, np.ndarray),
                            time_idx: int,
                            normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo discount process.

        The pseudo discount process is really -int_t^{t+dt} z_u du.

        Args:
            rate_spot: Pseudo short rate at time corresponding to time
                index.
            time_idx: Time index.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Incremented pseudo discount process.
        """
        mean = - rate_spot * self.discount_mean[time_idx][0]
        variance = self.discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def paths(self,
              spot: float,
              n_paths: int,
              rng: np.random.Generator = None,
              seed: int = None,
              antithetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo paths using exact discretization.

        Args:
            spot: Pseudo short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of pseudo short rate and discount processes
            represented on event_grid.
        """
        rate = np.zeros((self.event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self.event_grid.size, n_paths))
        if rng is None:
            rng = np.random.default_rng(seed)
        for time_idx in range(1, self.event_grid.size):
            correlation = self._correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, rng, antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self._rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self._discount_increment(rate[time_idx - 1], time_idx,
                                           x_discount)
        # Get pseudo discount factors on event_grid.
        discount = np.exp(discount)

        # Adjust for Pelsser transformation. TODO: Better explanation
        discount = discount.transpose() / np.exp(self.discount_variance / 2)

        return rate, discount.transpose()
