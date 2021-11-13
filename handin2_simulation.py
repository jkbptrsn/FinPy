import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import models.bachelier.call as ba_call
import models.black_scholes.call as bs_call
import mc_methods.standard_mc as mc
import handin2_utils as utils

rate = 0
vol = 0.2
strike = 1
expiry = 1

option = ba_call.Call(vol, strike, expiry)
# option = bs_call.Call(rate, vol, strike, expiry)

#############
# Figure 2a #
#############
n_prices = 10000
# Data set
spot, payoff, delta = utils.data_set(option, 0, n_prices)
plt.plot(spot, payoff, 'ok', markersize=5, markerfacecolor='none', alpha=0.2)
# Analytic result
spot_range = 0.02 * np.array(range(1, 100))
plt.plot(spot_range, option.price(spot_range, 0), 'k')
# Regression
poly_order = 7
poly_price, poly_delta = \
    utils.polynomials(poly_order, spot_range, spot, payoff, delta, w=1)
plt.plot(spot_range, poly_price, 'r')
plt.tick_params(labelsize=16)
plt.title("Figure 2a", fontsize=16)
plt.xlabel('Stock price', fontsize=16)
plt.ylabel('Call value', fontsize=16)
plt.xlim((0.25, 1.75))
plt.ylim((-0.05, 1.05))
plt.show()

#############
# Figure 2b #
#############
# Analytic result
plt.plot(spot_range, option.delta(spot_range, 0), 'k')
# Regression
plt.plot(spot_range, poly_delta, 'r')
plt.xlim((0.25, 1.75))
plt.ylim((-0.05, 1.05))
plt.tick_params(labelsize=16)
plt.title("Figure 2b", fontsize=16)
plt.xlabel('Stock price', fontsize=16)
plt.ylabel('Call delta', fontsize=16)
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
poly_price, poly_delta = utils.polynomials(poly_order, spot_range, spot, payoff, delta, w=0.01)
plt.plot(spot_range, poly_delta, 'y')
plt.xlim((0.25, 1.75))
plt.ylim((-0.15, 1.25))
plt.tick_params(labelsize=16)
plt.title("Figure 3", fontsize=16)
plt.xlabel('Stock price', fontsize=16)
plt.ylabel('Call delta', fontsize=16)
plt.show()
