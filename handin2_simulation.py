import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import models.bachelier.call as bachelier
import models.black_scholes.call as black_scholes


def data_set(option, time, n_paths):
    """Generate data set"""
    # Random spot values, independent of model
    spot = option.strike \
        + option.vol * math.sqrt(option.expiry) * norm.rvs(size=n_paths)
    spot_moved = option.path(spot, option.expiry - time, n_paths)
    payoff = option.payoff(spot_moved)

    # Path-wise "delta" values
    if option.model_name == 'Bachelier':
        delta = (spot_moved > option.strike) * 1
    elif option.model_name == 'Black-Scholes':
        delta = (spot_moved > option.strike) * spot_moved / spot

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


def discrete_hedging(
        option, n_paths, n_steps, mode='analytic', poly_order=7, w=1):
    """Discrete hedging..."""
    # Initial spot is 1 in all cases!!!
    spot = np.ones(n_paths)
    time_step = option.expiry / n_steps
    V = option.price(spot, 0)
    if mode == 'analytic':
        a = option.delta(spot, 0)
    elif mode == 'regression':
        spot_temp, payoff, delta = data_set(option, expiry, n_paths)
        poly_price, poly_delta = \
            polynomials(poly_order, spot, spot_temp, payoff, delta, w=w)
        a = poly_delta
    b = V - a * spot
    V_0 = V
    for n in range(1, n_steps):
        spot = option.path(spot, time_step, n_paths)
        V = a * spot + b
        if mode == 'analytic':
            a = option.delta(spot, n * time_step)
        elif mode == 'regression':
            spot_temp, payoff, delta = \
                data_set(option, expiry - n * time_step, n_paths)
            poly_price, poly_delta = \
                polynomials(poly_order, spot, spot_temp, payoff, delta, w=w)
            a = poly_delta
        b = V - a * spot
    spot = option.path(spot, time_step, n_paths)
    V = a * spot + b
    error = (V - np.maximum(spot - strike, 0)) / V_0
    error_mean = sum(error) / n_paths
    error_std = math.sqrt(sum((error - error_mean) ** 2) / n_paths)
    return error_mean, error_std


rate = 0
vol = 0.2
strike = 1
expiry = 1

# Call object
call = bachelier.Call(vol, strike, expiry)
# call = black_scholes.Call(rate, vol, strike, expiry)

#############
# Figure 2a #
#############
n_prices = 10000
# Data set
spot, payoff, delta = data_set(call, 0, n_prices)
plt.plot(spot, payoff, '.')
# Analytic result
spot_range = 0.02 * np.array(range(1, 100))
plt.plot(spot_range, call.price(spot_range, 0), 'k')
# Regression
poly_order = 7
poly_price, poly_delta = \
    polynomials(poly_order, spot_range, spot, payoff, delta, w=1)
plt.plot(spot_range, poly_price, 'r')
plt.xlabel('Stock price')
plt.ylabel('Call value')
plt.xlim((0.2, 1.8))
plt.ylim((-0.1, 1.1))
plt.show()

#############
# Figure 2b #
#############
# Analytic result
plt.plot(spot_range, call.delta(spot_range, 0), 'k')
# Regression
plt.plot(spot_range, poly_delta, 'r')
plt.xlim((0.2, 1.8))
plt.ylim((-0.15, 1.15))
plt.xlabel('Stock price')
plt.ylabel('Call delta')
plt.show()

############
# Figure 3 #
############
# Data set
plt.plot(spot, delta, '.')
# Analytical result
plt.plot(spot_range, call.delta(spot_range, 0), 'k')
# Regression, Price only
poly_price, poly_delta = polynomials(poly_order, spot_range, spot, payoff, delta, w=1)
plt.plot(spot_range, poly_delta, 'b')
# Regression, Price-Delta
poly_price, poly_delta = polynomials(poly_order, spot_range, spot, payoff, delta, w=0.5)
plt.plot(spot_range, poly_delta, 'r')
# Regression, Delta "only"
poly_price, poly_delta = polynomials(poly_order, spot_range, spot, payoff, delta, w=0.01)
plt.plot(spot_range, poly_delta, 'y')
plt.xlim((0.2, 1.8))
plt.ylim((-0.25, 1.25))
plt.xlabel('Stock price')
plt.ylabel('Call delta')
plt.show()

############
# Figure 4 #
############
n_batches = 10
n_hedge_paths = 1000
n_time_steps = 52

# Hedging with true delta
batch_mean = 0
batch_std = 0
for _ in range(n_batches):
    error_mean, error_std \
        = discrete_hedging(call, n_hedge_paths, n_time_steps)
    batch_mean += error_mean
    batch_std += error_std
batch_mean /= n_batches
batch_std /= n_batches
print(batch_mean, batch_std)

# Regressions
w_vector = [1.0, 0.5]
poly_vector = range(3, 10)
n_sim_vector = [250, 500] + list(range(1000, 7000, 1000))

hedge_error = \
    np.zeros((len(n_sim_vector) * len(w_vector) * len(poly_vector), n_batches))
for b in range(n_batches):
    counter = 0
    for n_sim in n_sim_vector:
        for w in w_vector:
            for p in poly_vector:
                # Relative hedge error
                mean, std = \
                    discrete_hedging(call, n_hedge_paths, n_time_steps,
                                     mode='regression', poly_order=p, w=w)
                hedge_error[counter, b] = std
                counter += 1
                print(n_sim, w, p)

counter = 0
for n_sim in n_sim_vector:
    for w in w_vector:
        for p in poly_vector:
            if math.fabs(w - 1.0) < 0.1:
                plt.plot(n_sim, sum(hedge_error[counter, :]) / n_batches, 'xy')
            else:
                plt.plot(n_sim, sum(hedge_error[counter, :]) / n_batches, '.b')
            counter += 1

plt.xlim((-100, 6100))
plt.ylim((0, 0.4))
plt.xlabel(f'#Simulations')
plt.ylabel('Standard deviation of relative hedge error')
plt.show()
