import math
import numpy as np

from models import sde
from utils import global_types
from utils import misc


class SDE(sde.SDE):
    """SDE for the short rate in the Vasicek model.

    The short rate r_t is defined by
        dr_t = kappa * (mean_rate - r_t) * dt + vol * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray):
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.event_grid = event_grid

        self.model_name = global_types.ModelName.VASICEK

        self.rate_mean = np.zeros((self.event_grid.size, 2))
        self.rate_variance = np.zeros(self.event_grid.size)
        self.discount_mean = np.zeros((self.event_grid.size, 2))
        self.discount_variance = np.zeros(self.event_grid.size)
        self.covariance = np.zeros(self.event_grid.size)

        self.initialization()

    def __repr__(self):
        return f"{self.model_name} SDE object"

    def initialization(self):
        """Initialization of the Monte-Carlo engine.

        Calculate time-dependent mean and variance of the short rate and
        discount processes, respectively.
        """
        self._calc_rate_mean()
        self._calc_rate_variance()
        self._calc_discount_mean()
        self._calc_discount_variance()
        self._calc_covariance()

    def _calc_rate_mean(self):
        """Conditional mean of short rate process.

        See Eq. (10.12), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        exp_kappa = np.exp(-self.kappa * np.diff(self.event_grid))
        self.rate_mean[0, 0] = 1
        self.rate_mean[1:, 0] = exp_kappa
        self.rate_mean[1:, 1] = self.mean_rate * (1 - exp_kappa)

    def _calc_rate_variance(self):
        """Conditional variance of short rate process.

        See Eq. (10.13), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        two_kappa = 2 * self.kappa
        exp_two_kappa = np.exp(-two_kappa * np.diff(self.event_grid))
        self.rate_variance[1:] = \
            self.vol ** 2 * (1 - exp_two_kappa) / two_kappa

    def _rate_increment(self,
                        spot: (float, np.ndarray),
                        time_idx: int,
                        normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment short rate process one time step.

        The spot rate is subtracted to get the increment.
        """
        mean = self.rate_mean[time_idx, 0] * spot + self.rate_mean[time_idx, 1]
        variance = self.rate_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand - spot

    def _calc_discount_mean(self):
        """Conditional mean of discount process.

        Here the discount process refers to -int_{t_1}^{t_2} r_t dt, see
        Eq. (10.12+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dt = np.diff(self.event_grid)
        exp_kappa = np.exp(-self.kappa * dt)
        exp_kappa = (1 - exp_kappa) / self.kappa
        self.discount_mean[1:, 0] = -exp_kappa
        self.discount_mean[1:, 1] = self.mean_rate * (exp_kappa - dt)

    def _calc_discount_variance(self):
        """Conditional variance of discount process.

        Here the discount process refers to -int_{t_1}^{t_2} r_t dt, see
        Eq. (10.13+), L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dt = np.diff(self.event_grid)
        vol_sq = self.vol ** 2
        exp_kappa = np.exp(-self.kappa * dt)
        two_kappa = 2 * self.kappa
        exp_two_kappa = np.exp(-two_kappa * dt)
        kappa_cubed = self.kappa ** 3
        self.discount_variance[1:] = \
            vol_sq * (4 * exp_kappa - 3 + two_kappa * dt
                      - exp_two_kappa) / (2 * kappa_cubed)

    def _discount_increment(self,
                            rate_spot: (float, np.ndarray),
                            time_idx: int,
                            normal_rand: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Increment discount process one time step."""
        mean = self.discount_mean[time_idx, 0] * rate_spot \
            + self.discount_mean[time_idx, 1]
        variance = self.discount_variance[time_idx]
        return mean + math.sqrt(variance) * normal_rand

    def _calc_covariance(self):
        """Covariance between between short rate and discount processes.

        See lemma 10.1.11, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dt = np.diff(self.event_grid)
        vol_sq = self.vol ** 2
        kappa_sq = self.kappa ** 2
        exp_kappa = np.exp(-self.kappa * dt)
        exp_two_kappa = np.exp(-2 * self.kappa * dt)
        self.covariance[1:] = \
            vol_sq * (2 * exp_kappa - exp_two_kappa - 1) / (2 * kappa_sq)

    def _correlation(self,
                     time_idx: int) -> float:
        """Correlation between short rate and discount processes."""
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
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of short rate and discount processes
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
        # Get discount factors on event_grid.
        discount = np.exp(discount)
        return rate, discount
