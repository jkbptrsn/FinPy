import math
import matplotlib.pyplot as plt
import numpy as np

import models.black_scholes.binary as bs_digital
import handin2_utils as utils

rate = 0
vol = 0.2
strike = 1
expiry = 1

# option = ba_call.Call(vol, strike, expiry)
# option = bs_call.Call(rate, vol, strike, expiry)
option = bs_digital.BinaryCashCall(rate, vol, strike, expiry)

# delta_mode = 'Path-wise'
delta_mode = 'LRM'

#############
# Figure 2a #
#############
n_prices = 10000
# Data set
spot, payoff, delta = utils.data_set(option, 0, n_prices, delta_mode=delta_mode)
plt.plot(spot, payoff, 'ok', markersize=5, markerfacecolor='none', alpha=0.2)
# Analytic result
spot_range = 0.02 * np.array(range(1, 150))
plt.plot(spot_range, option.price(spot_range, 0), 'k')
# Regression
poly_order = 7
poly_price, poly_delta = \
    utils.polynomials(poly_order, spot_range, spot, payoff, delta, w=1)
plt.plot(spot_range, poly_price, 'r')
plt.tick_params(labelsize=14)
plt.title("Sub-figure A", fontsize=14)
plt.xlabel('Stock price', fontsize=14)
plt.ylabel('Call value', fontsize=14)
plt.xlim((0., 2.5))
plt.ylim((-0.15, 1.25))
plt.show()

#############
# Figure 2b #
#############
# Analytic result
plt.plot(spot_range, option.delta(spot_range, 0), 'k')
plt.plot(spot_range, poly_delta, 'r')
plt.tick_params(labelsize=14)
plt.title("Sub-figure B", fontsize=14)
plt.xlabel('Stock price', fontsize=14)
plt.ylabel('Call delta', fontsize=14)
plt.xlim((0., 2.5))
plt.ylim((-0.4, 2.5))
plt.show()

############
# Figure 3 #
############
# Data set
plt.plot(spot, delta, 'ok', markersize=5, markerfacecolor='none', alpha=0.2)
# Analytical result
plt.plot(spot_range, option.delta(spot_range, 0), 'k')
# Regression, Price only
poly_price, poly_delta = utils.polynomials(poly_order, spot_range, spot, payoff, delta, w=1)
plt.plot(spot_range, poly_delta, 'b')
# Regression, Price-Delta
poly_price, poly_delta = utils.polynomials(poly_order, spot_range, spot, payoff, delta, w=0.5)
plt.plot(spot_range, poly_delta, 'r')
# Regression, Delta "only"
poly_price, poly_delta = utils.polynomials(poly_order, spot_range, spot, payoff, delta, w=0.001)
plt.plot(spot_range, poly_delta, 'y')
plt.tick_params(labelsize=14)
plt.title("Sub-figure C", fontsize=14)
plt.xlabel('Stock price', fontsize=14)
plt.ylabel('Call delta', fontsize=14)
plt.xlim((0., 2.5))
plt.ylim((-0.4, 2.5))
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
    mean, std = utils.discrete_hedging(option, n_hedge_paths, n_hedge_paths, n_time_steps)
    batch_mean += mean
    batch_std += std
batch_mean /= n_batches
batch_std /= n_batches
print(batch_mean, batch_std)

# Hedging with regression
w_vector = [1.0, 0.5]
poly_vector = range(3, 10, 1)

n_sim_vector = [250, 500] + list(range(1000, 7000, 1000))
# n_sim_vector = [1000, 2500] + list(range(5000, 35001, 10000))

hedge_error = \
    np.zeros((len(n_sim_vector) * len(w_vector) * len(poly_vector), n_batches))

for n in n_sim_vector:
    mean, std = \
        utils.discrete_hedging(option, n_hedge_paths, n, n_time_steps,
                               mode='regression', poly_order=8, w=1,
                               delta_mode=delta_mode)
    print(mean, std)

for b in range(n_batches):
    counter = 0
    for n_sim in n_sim_vector:
        for w in w_vector:
            for p in poly_vector:
                # Relative hedge error
                mean, std = \
                    utils.discrete_hedging(option, n_hedge_paths, n_sim,
                                           n_time_steps, mode='regression',
                                           poly_order=p, w=w,
                                           delta_mode=delta_mode)
                hedge_error[counter, b] = std
                counter += 1
                print(n_sim, w, p, std)

price_only = np.zeros((len(poly_vector), len(n_sim_vector)))
price_delta = np.zeros((len(poly_vector), len(n_sim_vector)))
counter = 0
for idx_s, n_sim in enumerate(n_sim_vector):
    for w in w_vector:
        for idx_p, p in enumerate(poly_vector):
            if math.fabs(w - 1.0) < 0.1:
                price_only[idx_p, idx_s] = sum(hedge_error[counter, :]) / n_batches
            else:
                price_delta[idx_p, idx_s] = sum(hedge_error[counter, :]) / n_batches
            counter += 1

for w in w_vector:
    for idx_p, p in enumerate(poly_vector):
        if math.fabs(w - 1.0) < 0.1:
            plt.plot(n_sim_vector, price_only[idx_p, :], '-xy', linewidth=1,
                     markersize=5, markerfacecolor='none', alpha=0.8)
        else:
            plt.plot(n_sim_vector, price_delta[idx_p, :], '-ob', linewidth=1,
                     markersize=5, markerfacecolor='none', alpha=0.8)

print(price_only)
print(price_delta)

hedge_sim = np.array(range(0, n_sim_vector[-1] + 1, 1000))
hedge_std = np.zeros(len(hedge_sim)) + batch_std
plt.plot(hedge_sim, hedge_std, '--r', linewidth=1)

plt.tick_params(labelsize=14)
plt.title("Sub-figure D", fontsize=14)
plt.xlabel('#simulations', fontsize=14)
plt.ylabel('Standard deviation of relative hedge error', fontsize=14)
plt.xlim((-100, 6100))
plt.ylim((0, 0.8))
plt.show()
