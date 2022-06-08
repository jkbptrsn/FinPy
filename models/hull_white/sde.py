import math
import numpy as np
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types
import utils.misc as misc


class SDE(sde.SDE):
    """Hull-White (Extended Vasicek) SDE for the short rate
        dx_t = (y_t - kappa_t * x_t) * dt + vol_t * dW_t,
    where
        r_t = x_t + f(0,t) (f is the instantaneous forward rate).
    Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

    - event_grid: event dates, i.e., trade date, payment dates,
                  extraordinary prepayment deadlines, etc.
    - int_step_size: integration step size (default is daily step size)

    IF KAPPA IS >1, THE INTEGRATION STEP SIZE SHOULD BE DECREASED!
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 forward_rate: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        self._kappa = kappa
        self._vol = vol
        self._forward_rate = forward_rate
        self._event_grid = event_grid
        self._int_step_size = int_step_size
        self._model_name = global_types.ModelName.HULL_WHITE

        self._rate_mean = np.zeros((event_grid.size, 2))
        self._rate_variance = np.zeros(event_grid.size)
        self._discount_mean = np.zeros((event_grid.size, 2))
        self._discount_variance = np.zeros(event_grid.size)
        self._covariance = np.zeros(event_grid.size)
        self._forward_rate_contrib = np.zeros((event_grid.size, 2))

        # Integration grid
        self._int_grid = None
        # Indices of event dates on integration grid
        self._int_event_idx = None

        # Arrays used in setting up the Monte-Carlo simulation
        self._kappa_int_grid = None
        self._vol_int_grid = None
        self._y_int_grid = None
        self._int_kappa_step = None

    def __repr__(self):
        return f"{self._model_name} SDE object"

    @property
    def kappa(self) -> misc.DiscreteFunc:
        return self._kappa

    @kappa.setter
    def kappa(self,
              kappa_: misc.DiscreteFunc):
        self._kappa = kappa_

    @property
    def vol(self) -> misc.DiscreteFunc:
        return self._vol

    @vol.setter
    def vol(self,
            vol_: misc.DiscreteFunc):
        self._vol = vol_

    @property
    def forward_rate(self) -> misc.DiscreteFunc:
        return self._forward_rate

    @forward_rate.setter
    def forward_rate(self,
                     forward_rate_: misc.DiscreteFunc):
        self._forward_rate = forward_rate_

    @property
    def model_name(self) -> global_types.ModelName:
        return self._model_name

###############################################################################

    @property
    def int_grid(self) -> np.ndarray:
        return self._int_grid

    @property
    def int_event_idx(self) -> np.ndarray:
        return self._int_event_idx

    @property
    def y_int_grid(self) -> np.ndarray:
        return self._y_int_grid

###############################################################################

    def initialization(self):
        """Initialize the Monte-Carlo simulation by calculating mean and
        variance of the short rate and discount processes, respectively.
        """
        self.integration_grid()
        self.kappa_vol_y()
        self.rate_mean()
        self.rate_variance()
        self.discount_mean()
        self.discount_variance()
        self.covariance()
        self.forward_rate_contrib()
        self._kappa_int_grid = None
        self._vol_int_grid = None
        self._y_int_grid = None
        self._int_kappa_step = None

    def integration_grid(self):
        """Time grid for integration of coefficients."""
        # Assume that the first event date is the initial time point on
        # the integration grid
        self._int_grid = np.array(self._event_grid[0])
        # The first event has index zero on the integration grid
        self._int_event_idx = np.array([0])
        # Step size between two adjacent event dates
        step_size_grid = np.diff(self._event_grid)
        for idx, step_size in enumerate(step_size_grid):
            # Number of integration steps
            steps = math.floor(step_size / self._int_step_size)
            initial_date = self._event_grid[idx]
            if steps == 0:
                grid = np.array([initial_date + step_size])
            else:
                grid = self._int_step_size * np.arange(1, steps + 1) \
                    + initial_date
                diff_step = step_size - steps * self._int_step_size
                if diff_step > 1.0e-8:
                    grid = np.append(grid, grid[-1] + diff_step)
            self._int_grid = np.append(self._int_grid, grid)
            self._int_event_idx = np.append(self._int_event_idx, grid.size)
        self._int_event_idx = np.cumsum(self._int_event_idx)

    def forward_rate_contrib(self):
        """Calculate contribution to short rate process and discount
        process from instantaneous forward rate.
        """
        self._forward_rate_contrib[:, 0] = \
            self._forward_rate.interpolation(self._event_grid)
        forward_rate = self._forward_rate.interpolation(self._int_grid)
        forward_rate_int = misc.trapz(self._int_grid, forward_rate)
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            # Slice of integration grid
            self._forward_rate_contrib[event_idx, 1] = \
                -np.sum(forward_rate_int[int_idx1:int_idx2 + 1])

    def kappa_vol_y(self):
        """Speed of mean reversion, volatility and y-function
        represented on integration grid.
        Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Speed of mean reversion interpolated on integration grid
        self._kappa_int_grid = self._kappa.interpolation(self._int_grid)
        # Volatility interpolated on integration grid
        self._vol_int_grid = self._vol.interpolation(self._int_grid)
        # Integration of speed of mean reversion using trapezoidal rule
        self._int_kappa_step = \
            np.append(np.array([0]),
                      misc.trapz(self._int_grid, self._kappa_int_grid))
        # Calculation of y-function on integration grid
        self._y_int_grid = np.zeros(self._int_grid.size)
        for idx in range(1, self._int_grid.size):
            # int_u^t kappa_t dt
            int_kappa = self._int_kappa_step[:idx + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            # Integrand in expression for y
            integrand = \
                np.exp(-2 * int_kappa) * self._vol_int_grid[:idx + 1] ** 2
            # Integration
            self._y_int_grid[idx] = \
                np.sum(misc.trapz(self._int_grid[:idx + 1], integrand))

    def rate_mean(self):
        """Factors for calculating conditional mean of short rate.
        Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2 + 1]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2 + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = \
                np.exp(-int_kappa) * self._y_int_grid[int_idx1:int_idx2 + 1]
            factor1 = math.exp(-np.sum(int_kappa))
            factor2 = np.sum(misc.trapz(int_grid, integrand))
            self._rate_mean[event_idx] = [factor1, factor2]

    def rate_variance(self):
        """Factors for calculating conditional variance of short rate.
        Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2 + 1]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2 + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = \
                np.exp(-int_kappa) * self._vol_int_grid[int_idx1:int_idx2 + 1]
            variance = np.sum(misc.trapz(int_grid, integrand))
            self._rate_variance[event_idx] = variance

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       time_idx: int,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment short rate process (the spot rate is subtracted to
        get the increment).
        """
        mean = \
            self._rate_mean[time_idx][0] * spot + self._rate_mean[time_idx][1]
        variance = self._rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def discount_mean(self):
        """Factors for calculating conditional mean of discount process,
        i.e., - int_t^{t+dt} r_u du.
        Eq. (10.42), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2 + 1]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2 + 1]
            # G-function
            # Eq. (10.18), L.B.G. Andersen & V.V. Piterbarg 2010
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            factor1 = np.sum(misc.trapz(int_grid, integrand))
            # Double time integral in Eq. (10.42)
            factor2 = np.array([0])
            for idx in range(int_idx1 + 1, int_idx2 + 1):
                int_grid_tmp = self._int_grid[int_idx1:idx + 1]
                int_kappa_tmp = self._int_kappa_step[int_idx1:idx + 1]
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self._y_int_grid[int_idx1:idx + 1]
                factor2 = \
                    np.append(factor2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            factor2 = np.sum(misc.trapz(int_grid, factor2))
            self._discount_mean[event_idx] = [factor1, factor2]

    def discount_variance(self):
        """Factors for calculating conditional variance of discount
        process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.43), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2 + 1]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2 + 1]
            # G-function
            # Eq. (10.18), L.B.G. Andersen & V.V. Piterbarg 2010
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            term1 = \
                self._y_int_grid[int_idx1] \
                * np.sum(misc.trapz(int_grid, integrand)) ** 2
            # Double time integral in Eq. (10.43)
            factor2 = np.array([0])
            for idx in range(int_idx1 + 1, int_idx2 + 1):
                int_grid_tmp = self._int_grid[int_idx1:idx + 1]
                int_kappa_tmp = self._int_kappa_step[int_idx1:idx + 1]
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self._y_int_grid[int_idx1:idx + 1]
                factor2 = \
                    np.append(factor2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            term2 = 2 * np.sum(misc.trapz(int_grid, factor2))
            self._discount_variance[event_idx] = term2 - term1

    def discount_increment(self,
                           rate_spot: (float, np.ndarray),
                           time_idx: int,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment discount process."""
        mean = \
            - rate_spot * self._discount_mean[time_idx][0] \
            - self._discount_mean[time_idx][1]
        variance = self._discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def covariance(self):
        """Covariance between between short rate and discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx]
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2 + 1]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2 + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            exp_kappa = np.exp(-int_kappa)
            cov = np.array([0])
            for idx in range(int_idx1 + 1, int_idx2 + 1):
                int_grid_temp = self._int_grid[int_idx1:idx + 1]
                int_kappa_temp = self._int_kappa_step[int_idx1:idx + 1]
                int_kappa_temp = np.cumsum(int_kappa_temp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_temp) \
                    * self._vol_int_grid[int_idx1:idx + 1] ** 2 \
                    * exp_kappa[:idx + 1 - int_idx1]
                cov = np.append(cov,
                                np.sum(misc.trapz(int_grid_temp, integrand)))
            self._covariance[event_idx] = - np.sum(misc.trapz(int_grid, cov))

    def correlation(self,
                    time_idx: int) -> float:
        """Correlation between between short rate and discount
        processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        covariance = self._covariance[time_idx]
        rate_var = self._rate_variance[time_idx]
        discount_var = self._discount_variance[time_idx]
        return covariance / math.sqrt(rate_var * discount_var)

    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        pass

    def path_time_grid(self,
                       spot: float,
                       time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def paths(self,
              spot: float,
              n_paths: int,
              seed: int = None,
              antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths(s), represented on _event_grid, of correlated
        short rate and discount processes using exact discretization.

        antithetic : Antithetic sampling for Monte-Carlo variance
        reduction. Defaults to False.
        """
        if antithetic and n_paths % 2 == 1:
            raise ValueError("In antithetic sampling, n_paths should be even.")
        rate = np.zeros((self._event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.ones((self._event_grid.size, n_paths))
        for time_idx in range(1, self._event_grid.size):
            correlation = self.correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, seed, antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self.rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self.discount_increment(rate[time_idx - 1], time_idx,
                                          x_discount)
        # Add forward rate contribution
        for time_idx in range(1, self._event_grid.size):
            rate[time_idx] += self._forward_rate_contrib[time_idx, 0]
            discount[time_idx] += self._forward_rate_contrib[time_idx, 1]
        return rate, discount
