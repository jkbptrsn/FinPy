import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

# Alphabetical order...


class DiscreteFunc:
    """Interpolation and extrapolation of a discrete function.

    Attributes:
        name: Name of the function.
        time_grid: Time grid in year fractions.
        values: Function value at each point on time grid.
        interp_scheme: Interpolation scheme.
            Default is flat interpolation ("zero"). Some other
            interpolation schemes are "linear", "quadratic", "cubic",
            etc. For more information, see the scipy documentation.
        extrap_scheme: Use corresponding extrapolation scheme. Default
            is True.
    """

    def __init__(self,
                 name: str,
                 time_grid: np.ndarray,
                 values: np.ndarray,
                 interp_scheme: str = "zero",
                 extrap_scheme: bool = True):
        self.name = name
        self.time_grid = time_grid
        self.values = values
        self.interp_scheme = interp_scheme
        self.extrap_scheme = extrap_scheme

    def interpolation(self,
                      interp_time_grid: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Interpolate (and extrapolate) on interp_time_grid.

        Args:
            interp_time_grid: Interpolation time grid in year fractions.

        Returns:
            Function values on interpolation time grid.
        """
        if self.extrap_scheme:
            extrap = "extrapolate"
        else:
            extrap = None
        f = interp1d(self.time_grid, self.values, kind=self.interp_scheme,
                     fill_value=extrap)
        return f(interp_time_grid)


def trapz(grid: np.ndarray,
          function: np.ndarray) -> np.ndarray:
    """Trapezoidal integration for each step along the grid."""
    dx = np.diff(grid)
    return dx * (function[1:] + function[:-1]) / 2


def cholesky_2d(correlation: float,
                n_sets: int,
                rng: np.random.Generator,
                antithetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Sets of two correlated standard normal random variables using
    Cholesky decomposition. In the 2-D case, the transformation is
    simply: (x1, correlation * x1 + sqrt{1 - correlation ** 2} * x2)
    """
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    corr_matrix = np.linalg.cholesky(corr_matrix)
    x1 = normal_realizations(n_sets, rng, antithetic)
    x2 = normal_realizations(n_sets, rng, antithetic)
    return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
        corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2


def normal_realizations(n_realizations: int,
                        rng: np.random.Generator,
                        antithetic: bool = False) -> np.ndarray:
    """Realizations of a standard normal random variable."""
    if antithetic and n_realizations % 2 == 1:
        raise ValueError("In antithetic sampling, the number of "
                         "realizations should be even.")
    anti = 1
    if antithetic:
        anti = 2
    realizations = rng.normal(size=n_realizations // anti)

#    realizations = norm.rvs(size=n_realizations // anti)

    if antithetic:
        realizations = np.append(realizations, -realizations)
    return realizations


def monte_carlo_error(realizations: np.ndarray) -> float:
    """Calculate the standard error of the Monte-Carlo estimate.

    Args:
        realizations: Realizations of the relevant random variable.

    Returns:
        Standard error.
    """
    sample_size = realizations.size
    sample_variance = realizations.var(ddof=1)
    return math.sqrt(sample_variance) / math.sqrt(sample_size)


def price_refinancing_bond(coupon: float,
                           n_payments: int,
                           sum_discount_factors: float) -> float:
    """Refinancing bond is an annuity. Return the difference between
    par and the price of the refinancing bond."""
    constant_payment = coupon / (1 - (1 + coupon) ** (-n_payments))
    return 1 - constant_payment * sum_discount_factors


def calc_refinancing_coupon(n_payments: int,
                            sum_discount_factors: float) -> float:
    """Calculate coupon of refinancing bond assuming par value."""
    arguments = (n_payments, sum_discount_factors)
    return brentq(price_refinancing_bond, -0.9, 0.9, args=arguments)
