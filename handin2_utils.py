import math
import numpy as np
from scipy.stats import norm


def data_set(option, time, n_paths, delta_mode="Path-wise"):
    """Generate data set"""
    # Random spot values

    # todo: should option.expiry be used here? Or time?
    spot = option.strike \
        + 1.7 * option.vol * math.sqrt(option.expiry) * norm.rvs(size=n_paths)

    spot_moved = option.path(spot, option.expiry - time, n_paths)
    # Payoff for each path
    payoff = option.payoff(spot_moved)
    # "Delta-value" for each path
    if delta_mode == "Path-wise":
        if option.model_name == 'Bachelier':
            delta = (spot_moved > option.strike) * 1
        elif option.model_name == 'Black-Scholes':
            delta = (spot_moved > option.strike) * spot_moved / spot

    # todo: double check the LRMs
    elif delta_mode == "LRM":
        if option.model_name == 'Bachelier':
            delta = option.payoff(spot_moved) \
                * (spot_moved - spot) / (option.vol ** 2 * (option.expiry - time))
        elif option.model_name == 'Black-Scholes':
            delta = option.payoff(spot_moved) \
                * (np.log(spot_moved / spot) + 0.5 * option.vol ** 2 * (option.expiry - time)) \
                / (option.vol ** 2 * (option.expiry - time) * spot)
    return spot, payoff, delta


def regression(poly_order, spot, payoff, delta, w):
    """Polynomial regression"""
    order = poly_order + 1
    x = np.ndarray((len(spot), order))
    for n in range(order):
        x[:, n] = spot ** n
    xt = x.transpose()
    y = np.ndarray((len(spot), order))
    y[:, 0] = 0 * spot
    for n in range(1, order):
        y[:, n] = n * spot ** (n - 1)
    yt = y.transpose()
    return np.matmul(
        np.linalg.inv(w * np.matmul(xt, x) + (1 - w) * np.matmul(yt, y)),
        w * np.matmul(xt, payoff) + (1 - w) * np.matmul(yt, delta))


def polynomials(poly_order, spot_range, spot, payoff, delta, w=1):
    """Construct polynomials"""
    theta = regression(poly_order, spot, payoff, delta, w)
    poly_price = spot_range * 0
    poly_delta = spot_range * 0
    for n in range(poly_order + 1):
        poly_price += theta[n] * spot_range ** n
    for n in range(poly_order):
        poly_delta += theta[n + 1] * (n + 1) * spot_range ** n
    return poly_price, poly_delta


def discrete_hedging(option, n_paths, n_data_points, n_steps, mode='analytic', poly_order=7,
                     w=1, delta_mode="Path-wise"):
    """Discrete hedging."""
    # Initial spot is 1 in all cases!!!
    spot = np.ones(n_paths)

    time_step = option.expiry / n_steps

    V = option.price(spot, 0)
    if mode == 'analytic':
        a = option.delta(spot, 0)
    elif mode == 'regression':
        spot_temp, payoff, delta = data_set(option, 0, n_data_points, delta_mode=delta_mode)
        poly_price, poly_delta = polynomials(poly_order, spot, spot_temp, payoff, delta, w=w)
        a = poly_delta
    b = V - a * spot
    V_0 = V
    for n in range(1, n_steps):
        spot = option.path(spot, time_step, n_paths)
        V = a * spot + b
        if mode == 'analytic':
            a = option.delta(spot, n * time_step)
        elif mode == 'regression':
            spot_temp, payoff, delta = data_set(option, n * time_step, n_data_points, delta_mode=delta_mode)
            poly_price, poly_delta = polynomials(poly_order, spot, spot_temp, payoff, delta, w=w)
            a = poly_delta
        b = V - a * spot
    spot = option.path(spot, time_step, n_paths)
    V = a * spot + b
    error = (V - option.payoff(spot)) / V_0
    error_mean = sum(error) / n_paths
    error_std = math.sqrt(sum((error - error_mean) ** 2) / n_paths)
    return error_mean, error_std
