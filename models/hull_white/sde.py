import math
import numpy as np
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types
import utils.misc as misc

# TODO: abstract base class for SDE


class SDEConstant(sde.SDE):
    """SDE for the pseudo short rate in the Hull-White (Extended
    Vasicek) model
        dx_t = (y_t - kappa * x_t) * dt + vol * dW_t,
    where
        r_t = x_t + f(0,t) (f is the instantaneous forward rate).
    Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    - kappa: Speed of mean reversion (constant)
    - vol: Volatility (constant)
    - discount_curve: Discount factor for each event date
    - event_grid: Event dates, i.e., trade date, payment dates, etc.
    - int_step_size: Integration/propagation step size
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.event_grid = event_grid
        self.int_step_size = int_step_size

        self.model_name = global_types.ModelName.HULL_WHITE_1F

        # Arrays used for exact discretization
        self.rate_mean = np.zeros((event_grid.size, 2))
        self.rate_variance = np.zeros(event_grid.size)
        self.discount_mean = np.zeros((event_grid.size, 2))
        self.discount_variance = np.zeros(event_grid.size)
        self.covariance = np.zeros(event_grid.size)

        # Integration grid
        self.int_grid = None
        # Indices of event dates on integration grid
        self.int_event_idx = None
        # y-function on integration and event grids
        # Eq. (10.17), L.B.G. Andersen & V.V. Piterbarg 2010.
        self.y_int_grid = None
        self.y_event_grid = np.zeros(event_grid.size)
        # Euler propagation scheme
        self.kappa_int_grid = None
        self.vol_int_grid = None
        # Array used in initialization of the SDE object
        self.int_kappa_step = None

    def __repr__(self):
        return f"{self.model_name} SDE object"

    def initialization(self):
        """Initialize the Monte-Carlo simulation:
        - Exact discretization: On event grid, calculate mean and
          variance of the short rate and discount processes,
          respectively.
        - Approximate discretization: On integration grid, ...
        """
        self.setup_int_grid()
        self.setup_kappa_vol_y()
        self.calc_rate_mean()
        self.calc_rate_variance()
        self.calc_discount_mean()
        self.calc_discount_variance()
        self.calc_covariance()
        # Array is not used after initialization
        self.int_kappa_step = None

    def setup_int_grid(self):
        """Time grid for numerical integration."""
        # Assume that the first event date is the initial time point on
        # the integration grid
        self.int_grid = np.array(self.event_grid[0])
        # The first event has index zero on the integration grid
        self.int_event_idx = np.array(0)
        # Step size between two adjacent event dates
        step_size_grid = np.diff(self.event_grid)
        for idx, step_size in enumerate(step_size_grid):
            # Number of integration steps
            steps = math.floor(step_size / self.int_step_size)
            initial_date = self.event_grid[idx]
            if steps == 0:
                grid = np.array(initial_date + step_size)
            else:
                grid = self.int_step_size * np.arange(1, steps + 1) \
                    + initial_date
                diff_step = step_size - steps * self.int_step_size
                # TODO: Is this necessary?
                if diff_step > 1.0e-8:
                    grid = np.append(grid, grid[-1] + diff_step)
            self.int_grid = np.append(self.int_grid, grid)
            self.int_event_idx = np.append(self.int_event_idx, grid.size)
        self.int_event_idx = np.cumsum(self.int_event_idx)

    def setup_kappa_vol_y(self):
        """Speed of mean reversion, volatility and y-function
        represented on integration grid.
        Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion on integration grid
        kappa = self.kappa.values[0]
        self.kappa_int_grid = kappa + 0 * self.int_grid
        # Volatility on integration grid
        vol = self.vol.values[0]
        self.vol_int_grid = vol + 0 * self.int_grid
        # Calculation of y-function on integration grid
        exp_kappa = np.exp(-2 * kappa * self.int_grid)
        self.y_int_grid = vol ** 2 * (1 - exp_kappa) / (2 * kappa)
        # Save y-function on event grid
        for idx, event_idx in enumerate(self.int_event_idx):
            self.y_event_grid[idx] = self.y_int_grid[event_idx]

    def calc_rate_mean(self):
        """Factors for calculating conditional mean of pseudo short
        rate.
        Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa = self.kappa.values[0]
        vol = self.vol.values[0]
        self.rate_mean[0] = [1, 0]
        self.rate_mean[1:, 0] = np.exp(-kappa * np.diff(self.event_grid))
        exp_kappa_1 = np.exp(-2 * kappa * self.event_grid[1:])
        exp_kappa_2 = np.exp(-kappa * np.diff(self.event_grid))
        event_grid_sum = self.event_grid[1:] + self.event_grid[:-1]
        exp_kappa_3 = np.exp(-kappa * event_grid_sum)
        self.rate_mean[1:, 1] = \
            vol ** 2 * (1 + exp_kappa_1 - exp_kappa_2 - exp_kappa_3) \
            / (2 * kappa ** 2)

    def calc_rate_variance(self):
        """Factors for calculating conditional variance of pseudo short
        rate.
        Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa = self.kappa.values[0]
        vol = self.vol.values[0]
        self.rate_variance[1:] = \
            vol ** 2 * (1 - np.exp(-2 * kappa * np.diff(self.event_grid))) \
            / (2 * kappa)

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       time_idx: int,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo short rate process (the spot value is
        subtracted to get the increment).
        """
        mean = self.rate_mean[time_idx][0] * spot + self.rate_mean[time_idx][1]
        variance = self.rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def calc_discount_mean(self):
        """Factors for calculating conditional mean of pseudo discount
        process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.42), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa = self.kappa.values[0]
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

    def calc_discount_variance(self):
        """Factors for calculating conditional variance of pseudo
        discount process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.43), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        self.discount_variance[1:] = \
            2 * self.discount_mean[1:, 1] \
            - self.y_event_grid[:-1] * self.discount_mean[1:, 0] ** 2

    def discount_increment(self,
                           rate_spot: (float, np.ndarray),
                           time_idx: int,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo discount process."""
        mean = \
            - rate_spot * self.discount_mean[time_idx][0] \
            - self.discount_mean[time_idx][1]
        variance = self.discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def calc_covariance(self):
        """Covariance between between pseudo short rate and pseudo
        discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa = self.kappa.values[0]
        vol = self.vol.values[0]
        exp_kappa_1 = np.exp(-2 * kappa * np.diff(self.event_grid))
        exp_kappa_2 = np.exp(-kappa * np.diff(self.event_grid))
        self.covariance[1:] = \
            -vol ** 2 * (1 + exp_kappa_1 - 2 * exp_kappa_2) / (2 * kappa ** 2)

    def correlation(self,
                    time_idx: int) -> float:
        """Correlation between between pseudo short rate and pseudo
        discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        covariance = self.covariance[time_idx]
        rate_var = self.rate_variance[time_idx]
        discount_var = self.discount_variance[time_idx]
        return covariance / math.sqrt(rate_var * discount_var)

    def paths(self,
              spot: float,
              n_paths: int,
              seed: int = None,
              antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths, represented on _event_grid, of correlated
        pseudo short rate and pseudo discount processes using exact
        discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        rate = np.zeros((self.event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self.event_grid.size, n_paths))
        if seed is not None:
            np.random.seed(seed)
        for time_idx in range(1, self.event_grid.size):
            correlation = self.correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, antithetic=antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self.rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self.discount_increment(rate[time_idx - 1], time_idx,
                                          x_discount)
        # Get pseudo discount factors at event dates
        discount = np.exp(discount)
        return rate, discount


class SDE(sde.SDE):
    """SDE for the pseudo short rate in the Hull-White (Extended
    Vasicek) model
        dx_t = (y_t - kappa_t * x_t) * dt + vol_t * dW_t,
    where
        r_t = x_t + f(0,t) (f is the instantaneous forward rate).
    Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    - kappa: Speed of mean reversion strip
    - vol: Volatility strip
    - discount_curve: Discount factor for each event date
    - event_grid: Event dates, i.e., trade date, payment dates, etc.
    - int_step_size: Integration/propagation step size

    Note: If kappa is >1, the integration step size should be decreased.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.event_grid = event_grid
        self.int_step_size = int_step_size

        self.model_name = global_types.ModelName.HULL_WHITE_1F

        # Arrays used for exact discretization
        self.rate_mean = np.zeros((event_grid.size, 2))
        self.rate_variance = np.zeros(event_grid.size)
        self.discount_mean = np.zeros((event_grid.size, 2))
        self.discount_variance = np.zeros(event_grid.size)
        self.covariance = np.zeros(event_grid.size)

        # Integration grid
        self.int_grid = None
        # Indices of event dates on integration grid
        self.int_event_idx = None
        # y-function on integration and event grids
        # Eq. (10.17), L.B.G. Andersen & V.V. Piterbarg 2010.
        self.y_int_grid = None
        self.y_event_grid = np.zeros(event_grid.size)
        # Euler propagation scheme
        self.kappa_int_grid = None
        self.vol_int_grid = None
        # Array used in initialization of the SDE object
        self.int_kappa_step = None

    def __repr__(self):
        return f"{self.model_name} SDE object"

    def initialization(self):
        """Initialize the Monte-Carlo simulation by calculating mean and
        variance of the short rate and discount processes, respectively.
        """
        self.setup_int_grid()
        self.setup_kappa_vol_y()
        self.calc_rate_mean()
        self.calc_rate_variance()
        self.calc_discount_mean()
        self.calc_discount_variance()
        self.calc_covariance()
        # Array is not used after initialization
        self.int_kappa_step = None

    def setup_int_grid(self):
        """Time grid for numerical integration."""
        # Assume that the first event date is the initial time point on
        # the integration grid
        self.int_grid = np.array(self.event_grid[0])
        # The first event has index zero on the integration grid
        self.int_event_idx = np.array(0)
        # Step size between two adjacent event dates
        step_size_grid = np.diff(self.event_grid)
        for idx, step_size in enumerate(step_size_grid):
            # Number of integration steps
            steps = math.floor(step_size / self.int_step_size)
            initial_date = self.event_grid[idx]
            if steps == 0:
                grid = np.array([initial_date + step_size])
            else:
                grid = self.int_step_size * np.arange(1, steps + 1) \
                    + initial_date
                diff_step = step_size - steps * self.int_step_size
                # TODO: Is this necessary?
                if diff_step > 1.0e-8:
                    grid = np.append(grid, grid[-1] + diff_step)
            self.int_grid = np.append(self.int_grid, grid)
            self.int_event_idx = np.append(self.int_event_idx, grid.size)
        self.int_event_idx = np.cumsum(self.int_event_idx)

    def setup_kappa_vol_y(self):
        """Speed of mean reversion, volatility and y-function
        represented on integration grid.
        Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion interpolated on integration grid
        self.kappa_int_grid = self.kappa.interpolation(self.int_grid)
        # Volatility interpolated on integration grid
        self.vol_int_grid = self.vol.interpolation(self.int_grid)
        # Integration of speed of mean reversion using trapezoidal rule
        self.int_kappa_step = \
            np.append(np.array(0),
                      misc.trapz(self.int_grid, self.kappa_int_grid))
        # Calculation of y-function on integration grid
        self.y_int_grid = np.zeros(self.int_grid.size)
        for idx in range(1, self.int_grid.size):
            # int_u^t_idx kappa_s ds
            int_kappa = self.int_kappa_step[:idx + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            # Integrand in expression for y
            integrand = \
                np.exp(-2 * int_kappa) * self.vol_int_grid[:idx + 1] ** 2
            # Integration
            self.y_int_grid[idx] = \
                np.sum(misc.trapz(self.int_grid[:idx + 1], integrand))
        # Save y-function on event grid
        for idx, event_idx in enumerate(self.int_event_idx):
            self.y_event_grid[idx] = self.y_int_grid[event_idx]

    def calc_rate_mean(self):
        """Factors for calculating conditional mean of pseudo short
        rate.
        Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        self.rate_mean[0] = [1, 0]
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self.int_event_idx[event_idx - 1]
            int_idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self.int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = np.append(self.int_kappa_step[int_idx1 + 1:int_idx2],
                                  np.array(0))
            factor1 = math.exp(-np.sum(int_kappa))
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = np.exp(-int_kappa) * self.y_int_grid[int_idx1:int_idx2]
            factor2 = np.sum(misc.trapz(int_grid, integrand))
            self.rate_mean[event_idx] = [factor1, factor2]

    def calc_rate_variance(self):
        """Factors for calculating conditional variance of pseudo short
        rate.
        Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self.int_event_idx[event_idx - 1]
            int_idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self.int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = np.append(self.int_kappa_step[int_idx1 + 1:int_idx2],
                                  np.array(0))
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = \
                np.exp(-int_kappa) * self.vol_int_grid[int_idx1:int_idx2]
            integrand = integrand ** 2
            variance = np.sum(misc.trapz(int_grid, integrand))
            self.rate_variance[event_idx] = variance

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       time_idx: int,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo short rate process (the spot value is
        subtracted to get the increment).
        """
        mean = self.rate_mean[time_idx][0] * spot + self.rate_mean[time_idx][1]
        variance = self.rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def calc_discount_mean(self):
        """Factors for calculating conditional mean of pseudo discount
        process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.42), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self.int_event_idx[event_idx - 1]
            int_idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self.int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self.int_kappa_step[int_idx1:int_idx2]
            # G-function in Eq. (10.18)
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            term1 = np.sum(misc.trapz(int_grid, integrand))
            # Double time integral in Eq. (10.42)
            term2 = np.array(0)
            for idx in range(int_idx1 + 1, int_idx2):
                int_grid_tmp = self.int_grid[int_idx1:idx + 1]
                int_kappa_tmp = \
                    np.append(self.int_kappa_step[int_idx1 + 1:idx + 1],
                              np.array(0))
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self.y_int_grid[int_idx1:idx + 1]
                term2 = \
                    np.append(term2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            term2 = np.sum(misc.trapz(int_grid, term2))
            self.discount_mean[event_idx] = [term1, term2]

    def calc_discount_variance(self):
        """Factors for calculating conditional variance of pseudo
        discount process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.43), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self.int_event_idx[event_idx - 1]
            int_idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self.int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self.int_kappa_step[int_idx1:int_idx2]
            # G-function in Eq. (10.18)
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            term1 = \
                self.y_int_grid[int_idx1] \
                * np.sum(misc.trapz(int_grid, integrand)) ** 2
            # Double time integral in Eq. (10.43)
            term2 = np.array(0)
            for idx in range(int_idx1 + 1, int_idx2):
                int_grid_tmp = self.int_grid[int_idx1:idx + 1]
                int_kappa_tmp = \
                    np.append(self.int_kappa_step[int_idx1 + 1:idx + 1],
                              np.array(0))
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self.y_int_grid[int_idx1:idx + 1]
                term2 = \
                    np.append(term2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            term2 = 2 * np.sum(misc.trapz(int_grid, term2))
            self.discount_variance[event_idx] = term2 - term1

    def discount_increment(self,
                           rate_spot: (float, np.ndarray),
                           time_idx: int,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo discount process."""
        mean = \
            - rate_spot * self.discount_mean[time_idx][0] \
            - self.discount_mean[time_idx][1]
        variance = self.discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def calc_covariance(self):
        """Covariance between between pseudo short rate and pseudo
        discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self.int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self.int_event_idx[event_idx - 1]
            int_idx2 = self.int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self.int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self.int_kappa_step[int_idx1:int_idx2]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            exp_kappa = np.exp(-int_kappa)
            cov = np.array(0)
            for idx in range(int_idx1 + 1, int_idx2):
                int_grid_temp = self.int_grid[int_idx1:idx + 1]
                int_kappa_temp = \
                    np.append(self.int_kappa_step[int_idx1 + 1:idx + 1],
                              np.array(0))
                int_kappa_temp = np.cumsum(int_kappa_temp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_temp) \
                    * self.vol_int_grid[int_idx1:idx + 1] ** 2 \
                    * exp_kappa[:idx + 1 - int_idx1]
                cov = np.append(cov,
                                np.sum(misc.trapz(int_grid_temp, integrand)))
            self.covariance[event_idx] = - np.sum(misc.trapz(int_grid, cov))

    def correlation(self,
                    time_idx: int) -> float:
        """Correlation between between pseudo short rate and pseudo
        discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        covariance = self.covariance[time_idx]
        rate_var = self.rate_variance[time_idx]
        discount_var = self.discount_variance[time_idx]
        return covariance / math.sqrt(rate_var * discount_var)

    def paths(self,
              spot: float,
              n_paths: int,
              seed: int = None,
              antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths, represented on _event_grid, of correlated
        pseudo short rate and pseudo discount processes using exact
        discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        rate = np.zeros((self.event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self.event_grid.size, n_paths))
        if seed is not None:
            np.random.seed(seed)
        for time_idx in range(1, self.event_grid.size):
            correlation = self.correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, antithetic=antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self.rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self.discount_increment(rate[time_idx - 1], time_idx,
                                          x_discount)
        # Get pseudo discount factors at event dates
        discount = np.exp(discount)
        return rate, discount
