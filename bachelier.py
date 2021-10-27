import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def bachelier_paths(spot, vol, time, n_paths):
    """Paths in Bachelier model"""
    return spot + vol * math.sqrt(time) * norm.rvs(0, 1, n_paths)


def bachelier_call_price(spot, strike, vol, expiry, time):
    """Price of European call option in Bachelier model"""
    factor1 = (spot - strike)
    factor2 = vol * math.sqrt(expiry - time)
    return factor1 * norm.cdf(factor1 / factor2) \
        + factor2 * norm.pdf(factor1 / factor2)


if __name__ == "__main__":
    vol = 0.2
    expiry = 1
    strike = 1

    n_paths = 10000
    spot_value = np.zeros(n_paths)
    call_price = np.zeros(n_paths)

    for n in range(n_paths):
        spot = strike + vol * math.sqrt(expiry) * norm.rvs()
        spot_moved = bachelier_paths(spot, vol, expiry, 1)
        spot_value[n] = spot
        call_price[n] = np.maximum(spot_moved - strike, 0)

    plt.plot(spot_value, call_price, '.')

    spot_range = 0.02 * np.array(range(100))
    plt.plot(spot_range, bachelier_call_price(spot_range, strike, vol, expiry, 0), 'k')

    plt.xlim((0.2, 1.8))
    plt.ylim((-0.05, 1.05))

    plt.show()
