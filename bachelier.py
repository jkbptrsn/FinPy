import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def paths(spot, vol, time, n_paths):
    """
    Paths in Bachelier model.

    Parameters
    ----------
    spot : float
    vol : float
    time : float
    n_paths : int

    Returns
    -------
    float
    """
    return spot + vol * norm.rvs(scale=math.sqrt(time), size=n_paths)


def call_price(spot, strike, vol, expiry, time):
    """
    Price of European call option in Bachelier model.

    Parameters
    ----------
    spot : float / np.array
    strike : float
    vol : float
    expiry : float
    time : float

    Returns
    -------
    float / np.array
    """
    factor1 = spot - strike
    factor2 = vol * math.sqrt(expiry - time)
    return factor1 * norm.cdf(factor1 / factor2) \
        + factor2 * norm.pdf(factor1 / factor2)


def call_delta(spot, strike, vol, expiry, time):
    """
    Delta of European call option in Bachelier model.

    Parameters
    ----------
    spot : float / np.array
    strike : float
    vol : float
    expiry : float
    time : float

    Returns
    -------
    float / np.array
    """
    factor1 = spot - strike
    factor2 = vol * math.sqrt(expiry - time)
    return norm.cdf(factor1 / factor2) \
        + factor1 * norm.pdf(factor1 / factor2) / factor2 \
        - factor1 * norm.pdf(factor1 / factor2) / factor2


if __name__ == "__main__":
    vol = 0.2
    expiry = 1
    strike = 1

    n_paths = 10000

    # MC data
    spot = strike + vol * norm.rvs(scale=math.sqrt(expiry), size=n_paths)
    spot_moved = np.array([paths(s, vol, expiry, 1) for s in spot])
    call = np.maximum(spot_moved - strike, 0)
    plt.plot(spot, call, '.')

    # Analytic result
    spot_range = 0.02 * np.array(range(100))
    plt.plot(spot_range, call_price(spot_range, strike, vol, expiry, 0), 'k')

    # Regression
    poly_order = 7
    poly_order += 1
    x_mat = np.ndarray((n_paths, poly_order))
    for p in range(poly_order):
        x_mat[:, p] = spot ** p
    x_mat_trans = x_mat.transpose()
    theta = \
        np.matmul(
            np.linalg.inv(
                np.matmul(x_mat_trans, x_mat)), np.matmul(x_mat_trans, call))
    poly_price = spot_range * 0
    poly_delta = spot_range * 0
    for p in range(poly_order):
        poly_price += theta[p] * spot_range ** p
    for p in range(poly_order - 1):
        poly_delta += theta[p + 1] * (p + 1) * spot_range ** p
    plt.plot(spot_range, poly_price, 'b')
    plt.xlim((0.2, 1.8))
    plt.ylim((-0.05, 1.05))
    plt.show()

    # Delta plot
    plt.plot(spot_range, call_delta(spot_range, strike, vol, expiry, 0), 'r')
    plt.plot(spot_range, poly_delta, 'b')
    plt.xlim((0.2, 1.8))
    plt.ylim((-0.15, 1.15))
    plt.show()
