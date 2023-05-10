import math
import numpy as np

from utils import global_types
from utils import misc


class SDEBasic:
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


class SDEConstant(SDEBasic):
    pass


class SDEPiecewise(SDEBasic):
    pass
