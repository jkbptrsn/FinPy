import math
import numpy as np
from typing import Tuple

import models.sde as sde
import utils.global_types as global_types
import utils.misc as misc


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
    - event_grid: Event dates, i.e., trade date, payment dates,
                  extraordinary prepayment deadlines, etc.
    - int_step_size: Integration step size (default is 1/365)

    Note: If kappa is >1, the integration step size should be decreased.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        self._kappa = kappa
        self._vol = vol
        self._discount_curve = discount_curve
        self._event_grid = event_grid
        self._int_step_size = int_step_size

        self._model_name = global_types.ModelName.HULL_WHITE_1F

        self._rate_mean = np.zeros((event_grid.size, 2))
        self._rate_variance = np.zeros(event_grid.size)
        self._discount_mean = np.zeros((event_grid.size, 2))
        self._discount_variance = np.zeros(event_grid.size)
        self._covariance = np.zeros(event_grid.size)

        # Integration grid
        self._int_grid = None
        # Indices of event dates on integration grid
        self._int_event_idx = None
        # y-function on event grid
        # Eq. (10.17), L.B.G. Andersen & V.V. Piterbarg 2010.
        self._y_event_grid = np.zeros(event_grid.size)

        # Arrays used in setting up the SDE object
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
    def discount_curve(self) -> misc.DiscreteFunc:
        return self._discount_curve

    @discount_curve.setter
    def discount_curve(self,
                       discount_curve_: misc.DiscreteFunc):
        self._discount_curve = discount_curve_

    @property
    def event_grid(self) -> np.ndarray:
        return self._event_grid

    @event_grid.setter
    def event_grid(self,
                   event_grid_: np.ndarray):
        self._event_grid = event_grid_

    @property
    def model_name(self) -> global_types.ModelName:
        return self._model_name

    @property
    def int_grid(self) -> np.ndarray:
        return self._int_grid

    @property
    def int_event_idx(self) -> np.ndarray:
        return self._int_event_idx

    @property
    def y_event_grid(self) -> np.ndarray:
        return self._y_event_grid

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
        # The following arrays are not used after initialization
        self._kappa_int_grid = None
        self._vol_int_grid = None
        self._y_int_grid = None
        self._int_kappa_step = None

    def integration_grid(self):
        """Time grid for numerical integration."""
        # Assume that the first event date is the initial time point on
        # the integration grid
        self._int_grid = np.array(self._event_grid[0])
        # The first event has index zero on the integration grid
        self._int_event_idx = np.array(0)
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
                # TODO: Is this necessary?
                if diff_step > 1.0e-8:
                    grid = np.append(grid, grid[-1] + diff_step)
            self._int_grid = np.append(self._int_grid, grid)
            self._int_event_idx = np.append(self._int_event_idx, grid.size)
        self._int_event_idx = np.cumsum(self._int_event_idx)

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
            np.append(np.array(0),
                      misc.trapz(self._int_grid, self._kappa_int_grid))
        # Calculation of y-function on integration grid
        self._y_int_grid = np.zeros(self._int_grid.size)
        for idx in range(1, self._int_grid.size):
            # int_u^t_idx kappa_s ds
            int_kappa = self._int_kappa_step[:idx + 1]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            # Integrand in expression for y
            integrand = \
                np.exp(-2 * int_kappa) * self._vol_int_grid[:idx + 1] ** 2
            # Integration
            self._y_int_grid[idx] = \
                np.sum(misc.trapz(self._int_grid[:idx + 1], integrand))
        # Save y-function on event grid
        for idx, event_idx in enumerate(self._int_event_idx):
            self._y_event_grid[idx] = self._y_int_grid[event_idx]

    def rate_mean(self):
        """Factors for calculating conditional mean of pseudo short
        rate.
        Eq. (10.40), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        self._rate_mean[0] = [1, 0]
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = np.append(self._int_kappa_step[int_idx1 + 1:int_idx2],
                                  np.array(0))
            factor1 = math.exp(-np.sum(int_kappa))
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = \
                np.exp(-int_kappa) * self._y_int_grid[int_idx1:int_idx2]
            factor2 = np.sum(misc.trapz(int_grid, integrand))
            self._rate_mean[event_idx] = [factor1, factor2]

    def rate_variance(self):
        """Factors for calculating conditional variance of pseudo short
        rate.
        Eq. (10.41), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = np.append(self._int_kappa_step[int_idx1 + 1:int_idx2],
                                  np.array(0))
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            integrand = \
                np.exp(-int_kappa) * self._vol_int_grid[int_idx1:int_idx2]
            integrand = integrand ** 2
            variance = np.sum(misc.trapz(int_grid, integrand))
            self._rate_variance[event_idx] = variance

    def rate_increment(self,
                       spot: (float, np.ndarray),
                       time_idx: int,
                       normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo short rate process (the spot value is
        subtracted to get the increment).
        """
        mean = \
            self._rate_mean[time_idx][0] * spot + self._rate_mean[time_idx][1]
        variance = self._rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def discount_mean(self):
        """Factors for calculating conditional mean of pseudo discount
        process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.42), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2]
            # G-function in Eq. (10.18)
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            term1 = np.sum(misc.trapz(int_grid, integrand))
            # Double time integral in Eq. (10.42)
            term2 = np.array(0)
            for idx in range(int_idx1 + 1, int_idx2):
                int_grid_tmp = self._int_grid[int_idx1:idx + 1]
                int_kappa_tmp = \
                    np.append(self._int_kappa_step[int_idx1 + 1:idx + 1],
                              np.array(0))
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self._y_int_grid[int_idx1:idx + 1]
                term2 = \
                    np.append(term2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            term2 = np.sum(misc.trapz(int_grid, term2))
            self._discount_mean[event_idx] = [term1, term2]

    def discount_variance(self):
        """Factors for calculating conditional variance of pseudo
        discount process, i.e., - int_t^{t+dt} r_u du.
        Eq. (10.43), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2]
            # G-function in Eq. (10.18)
            int_kappa = np.cumsum(int_kappa)
            integrand = np.exp(-int_kappa)
            term1 = \
                self._y_int_grid[int_idx1] \
                * np.sum(misc.trapz(int_grid, integrand)) ** 2
            # Double time integral in Eq. (10.43)
            term2 = np.array(0)
            for idx in range(int_idx1 + 1, int_idx2):
                int_grid_tmp = self._int_grid[int_idx1:idx + 1]
                int_kappa_tmp = \
                    np.append(self._int_kappa_step[int_idx1 + 1:idx + 1],
                              np.array(0))
                int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
                integrand = \
                    np.exp(-int_kappa_tmp) * self._y_int_grid[int_idx1:idx + 1]
                term2 = \
                    np.append(term2,
                              np.sum(misc.trapz(int_grid_tmp, integrand)))
            term2 = 2 * np.sum(misc.trapz(int_grid, term2))
            self._discount_variance[event_idx] = term2 - term1

    def discount_increment(self,
                           rate_spot: (float, np.ndarray),
                           time_idx: int,
                           normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment pseudo discount process."""
        mean = \
            - rate_spot * self._discount_mean[time_idx][0] \
            - self._discount_mean[time_idx][1]
        variance = self._discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def covariance(self):
        """Covariance between between pseudo short rate and pseudo
        discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        for event_idx in range(1, self._int_event_idx.size):
            # Integration indices of two adjacent event dates
            int_idx1 = self._int_event_idx[event_idx - 1]
            int_idx2 = self._int_event_idx[event_idx] + 1
            # Slice of integration grid
            int_grid = self._int_grid[int_idx1:int_idx2]
            # Slice of time-integrated kappa for each integration step
            int_kappa = self._int_kappa_step[int_idx1:int_idx2]
            int_kappa = np.cumsum(int_kappa[::-1])[::-1]
            exp_kappa = np.exp(-int_kappa)
            cov = np.array(0)
            for idx in range(int_idx1 + 1, int_idx2):
                int_grid_temp = self._int_grid[int_idx1:idx + 1]
                int_kappa_temp = \
                    np.append(self._int_kappa_step[int_idx1 + 1:idx + 1],
                              np.array(0))
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
        """Correlation between between pseudo short rate and pseudo
        discount processes.
        Lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        covariance = self._covariance[time_idx]
        rate_var = self._rate_variance[time_idx]
        discount_var = self._discount_variance[time_idx]
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
        rate = np.zeros((self._event_grid.size, n_paths))
        rate[0, :] = spot
        discount = np.zeros((self._event_grid.size, n_paths))
        for time_idx in range(1, self._event_grid.size):
            correlation = self.correlation(time_idx)
            x_rate, x_discount = \
                misc.cholesky_2d(correlation, n_paths, seed, antithetic)
            rate[time_idx] = rate[time_idx - 1] \
                + self.rate_increment(rate[time_idx - 1], time_idx, x_rate)
            discount[time_idx] = discount[time_idx - 1] \
                + self.discount_increment(rate[time_idx - 1], time_idx,
                                          x_discount)
        # Get pseudo discount factors at event dates
        discount = np.exp(discount)
        return rate, discount
