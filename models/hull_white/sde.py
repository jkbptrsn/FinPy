import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types


class Function:
    """Time-dependent function defined on time_grid with interpolation
    and extrapolation schemes.
    """

    def __init__(self,
                 name: str,
                 time_grid: np.ndarray,
                 values: np.ndarray,
                 interpolation: str = "zero",
                 extrapolation: bool = True):
        self._name = name
        self._time_grid = time_grid
        self._values = values
        self._interpolation = interpolation
        self._extrapolation = extrapolation

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self,
             name_: str):
        self._name = name_

    @property
    def time_grid(self) -> np.ndarray:
        return self._time_grid

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def interp_scheme(self) -> str:
        return self._interpolation

    @interp_scheme.setter
    def interp_scheme(self, interpolation_: str):
        self._interpolation = interpolation_

    def interpolation(self,
                      time_grid_new: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Interpolate (and extrapolate) on time_grid_new."""
        if self._extrapolation:
            fill_value = "extrapolate"
        else:
            fill_value = ""
        f = interp1d(self._time_grid, self._values,
                     kind=self._interpolation, fill_value=fill_value)
        return f(time_grid_new)

    def integration(self,
                    time_grid_new: (float, np.ndarray)) -> float:
        """Flat integration on time_grid_new. For each sub-interval, the
        function value at the left side is used.
        """
        dt = time_grid_new[1:] - time_grid_new[:-1]
        return np.sum(dt * self.interpolation(time_grid_new[:-1]))


class SDE(sde.SDE):
    """Hull-White (Extended Vasicek) SDE for the short rate
        dx_t = y_t - kappa_t * x_t) * dt + vol_t * dW_t,
    where
        r_t = x_t + f(0,t) (instantaneous forward rate).

    - event_grid: event dates (payments, deadlines, etc.)
    - int_step: integration step size between event dates
    """

    def __init__(self,
                 kappa: Function,
                 vol: Function,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 11): # 1 / 365
        self._kappa = kappa
        self._vol = vol
        self._event_grid = event_grid
        self._int_step_size = int_step_size

        self._rate_mean = np.zeros(event_grid.size)
        self._rate_variance = np.zeros(event_grid.size)
        self._discount_mean = np.zeros(event_grid.size)
        self._discount_variance = np.zeros(event_grid.size)
        self._covariance = np.zeros(event_grid.size)

        self._int_grid = np.empty(0)
        self._int_event_idx = np.empty(0)
        self._model_name = global_types.ModelName.HULL_WHITE

    def __repr__(self):
        return f"{self._model_name} SDE object"

    @property
    def kappa(self) -> Function:
        return self._kappa

    @kappa.setter
    def kappa(self,
              kappa_: Function):
        self._kappa = kappa_

    @property
    def vol(self) -> Function:
        return self._vol

    @vol.setter
    def vol(self,
            vol_: Function):
        self._vol = vol_

    @property
    def int_grid(self) -> np.ndarray:
        return self._int_grid

    @property
    def int_event_idx(self) -> np.ndarray:
        return self._int_event_idx

    @property
    def model_name(self) -> global_types.ModelName:
        return self._model_name

    def initialization(self):
        """Initialize Monte-Carlo simulation..."""
        self.integration_grid()
        self.rate_mean()
        self.rate_variance()

    def integration_grid(self):
        """..."""
        for idx, event_date in enumerate(self._event_grid):
            if idx == 0:
                step_size = event_date
                initial_date = 0
                subtract_index = 1
            else:
                step_size = self._event_grid[idx] - self._event_grid[idx - 1]
                initial_date = self._event_grid[idx - 1]
                subtract_index = 0
            steps = math.floor(step_size / self._int_step_size)
            if steps == 0:
                grid = step_size * np.arange(1, 2) + initial_date
            else:
                grid = self._int_step_size * np.arange(1, steps + 1) \
                    + initial_date
                diff_step = step_size - steps * self._int_step_size
                if diff_step > 1.0e-6:
                    grid = np.append(grid, grid[-1] + diff_step)
            self._int_grid = np.append(self._int_grid, grid)
            self._int_event_idx = \
                np.append(self._int_event_idx, grid.size - subtract_index)
        self._int_event_idx = np.int_(np.cumsum(self._int_event_idx))

    @staticmethod
    def trapezoidal(grid: np.ndarray,
                    function: np.ndarray) -> np.ndarray:
        """Trapezoidal integration for each step..."""
        dx = grid[1:] - grid[:-1]
        return dx * (function[1:] + function[:-1]) / 2

    def rate_mean(self):
        """Conditional mean factors...
        Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa = self._kappa.interpolation(self._int_grid)
        vol = self._vol.interpolation(self._int_grid)

        int_kappa_step = self.trapezoidal(self._int_grid, kappa)

        # Calculation of y-function on integration grid
        y = np.zeros(self._int_grid.size)
        for idx in range(self._int_grid.size):
            grid_temp = self._int_grid[:idx + 1]
            int_kappa_temp = int_kappa_step[:idx + 1]
            vol_temp = vol[:idx + 1]
#            vol_temp = (vol_temp[1:] + vol_temp[:-1]) / 2
            int_kappa = np.cumsum(int_kappa_temp[::-1])[::-1]
            integrand = np.exp(- 2 * int_kappa) * vol_temp ** 2
            y[idx] = np.sum(self.trapezoidal(grid_temp, integrand))

        for event_idx in range(1, self._int_event_idx.size):

            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            grid_temp = self._int_grid[int_idx1:int_idx2 + 1]

            int_kappa_temp = int_kappa_step[int_idx1:int_idx2 + 1]
            mean_factor1 = math.exp(-np.sum(int_kappa_temp))

            int_kappa = np.cumsum(int_kappa_temp[::-1])[::-1]
            y_temp = y[int_idx1:int_idx2 + 1]
            integrand = np.exp(- int_kappa) * y_temp
            mean_factor2 = np.sum(self.trapezoidal(grid_temp, integrand))

            mean_factors = np.array(mean_factor1, mean_factor2)
            self._rate_mean = np.append(self._rate_mean, mean_factors)

    def rate_variance(self):
        """Conditional variance factor...
        Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa = self._kappa.interpolation(self._int_grid)
        vol = self._vol.interpolation(self._int_grid)

        int_kappa_step = self.trapezoidal(self._int_grid, kappa)

        for event_idx in range(1, self._int_event_idx.size):

            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            grid_temp = self._int_grid[int_idx1:int_idx2 + 1]

            int_kappa_temp = int_kappa_step[int_idx1:int_idx2 + 1]
            int_kappa = np.cumsum(int_kappa_temp[::-1])[::-1]

            integrand = np.exp(- 2 * int_kappa) * vol ** 2
            variance = np.sum(self.trapezoidal(grid_temp, integrand))

            self._rate_variance = np.append(self._rate_variance, variance)

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       dt: float,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment short rate process (the spot rate is subtracted to
        get the increment).
        """
        return self.rate_mean(spot, dt) \
            + math.sqrt(self.rate_variance(dt)) * normal_rand - spot

    def discount_mean(self,
                      rate_spot: (float, np.ndarray),
                      dt: float) -> (float, np.ndarray):
        """Conditional mean of discount process, i.e.,
        - int_t^{t+dt} r_u du.
        Eq. (10.12+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        exp_kappa = math.exp(- self._kappa * dt)
        return - self._mean_rate * dt \
            - (rate_spot - self._mean_rate) * (1 - exp_kappa) / self._kappa

    def discount_variance(self,
                          dt: float) -> float:
        """Conditional variance of discount process, i.e.,
        - int_t^{t+dt} r_u du.
        Eq. (10.13+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        vol_sq = self._vol ** 2
        exp_kappa = math.exp(- self._kappa * dt)
        two_kappa = 2 * self._kappa
        exp_two_kappa = math.exp(- two_kappa * dt)
        kappa_cubed = self._kappa ** 3
        return vol_sq * (4 * exp_kappa - 3 + two_kappa * dt
                         - exp_two_kappa) / (2 * kappa_cubed)

    def discount_increment(self,
                           rate_spot: (float, np.ndarray),
                           time: float,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment discount process."""
        return self.discount_mean(rate_spot, time) \
            + np.sqrt(self.discount_variance(time)) * normal_rand

    def covariance(self,
                   dt: float) -> float:
        """Covariance between between short rate and discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        vol_sq = self._vol ** 2
        kappa_sq = self._kappa ** 2
        exp_kappa = math.exp(- self._kappa * dt)
        exp_two_kappa = math.exp(- 2 * self._kappa * dt)
        return vol_sq * (2 * exp_kappa - exp_two_kappa - 1) / (2 * kappa_sq)

    def correlation(self,
                    dt: float) -> float:
        """Correlation between between short rate and discount
        processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        covariance = self.covariance(dt)
        rate_var = self.rate_variance(dt)
        discount_var = self.discount_variance(dt)
        return covariance / math.sqrt(rate_var * discount_var)

    def cholesky(self,
                 dt: float,
                 n_paths: int) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):
        """Correlated standard normal random variables using Cholesky
        decomposition. In the 2-D case, the transformation is simply:
        x1, correlation * x1 + sqrt(1 - correlation ** 2) * x2
        """
        correlation = self.correlation(dt)
        corr_matrix = np.array([[1, correlation], [correlation, 1]])
        corr_matrix = np.linalg.cholesky(corr_matrix)
        x1 = norm.rvs(size=n_paths)
        x2 = norm.rvs(size=n_paths)
        return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
            corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2

    def path(self,
             spot: (float, np.ndarray),
             dt: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate paths(s), during time step dt, of correlated short
        rate and discount processes using exact discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        if antithetic:
            if n_paths % 2 == 1:
                raise ValueError("In antithetic sampling, "
                                 "n_paths should be even.")
            x_rate, x_discount = self.cholesky(dt, n_paths // 2)
            x_rate = np.append(x_rate, -x_rate)
            x_discount = np.append(x_discount, -x_discount)
        else:
            x_rate, x_discount = self.cholesky(dt, n_paths)
        return self.rate_increment(spot, dt, x_rate), \
            self.discount_increment(spot, dt, x_discount)

    def path_time_grid(self,
                       spot: float,
                       time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one path, represented on time_grid, correlated short
        rate and discount processes using exact discretization.
        """
        delta_t = time_grid[1:] - time_grid[:-1]
        rate = np.zeros(delta_t.shape[0] + 1)
        rate[0] = spot
        discount = np.zeros(delta_t.shape[0] + 1)
        for count, dt in enumerate(delta_t):
            x_rate, x_discount = self.cholesky(dt, 1)
            rate[count + 1] = rate[count] \
                + self.rate_increment(rate[count], dt, x_rate)
            discount[count + 1] = discount[count] \
                + self.discount_increment(rate[count], dt, x_discount)
        return rate, np.exp(discount)
