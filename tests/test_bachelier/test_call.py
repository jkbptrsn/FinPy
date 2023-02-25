import math
import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.bachelier.sde as sde
import models.bachelier.call as call
import models.bachelier.put as put


class CallOption(unittest.TestCase):

    def test_something(self):
        self.assertTrue(1 < 2)


if __name__ == '__main__':

    # Plot Bachelier scenarios
    rate = 0.05
    vol = 10.0
    time = 0
    time_idx = 0
    expiry = 2
    expiry_idx = 1000
    event_grid = expiry * np.array(range(expiry_idx + 1)) / expiry_idx
    bachelier = sde.SDE(rate, vol, event_grid)
    bachelier.initialization()
    spot = 50
    n_paths = 10
    seed = 0
    paths = bachelier.paths(spot, n_paths, seed)
    for n in range(n_paths):
        plt.plot(event_grid, paths[:, n])
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.show()

    # Plot call and put option prices
    strike = 50
    expiry_idx = 10
    event_grid = expiry * np.array(range(expiry_idx + 1)) / expiry_idx
    bachelier = sde.SDE(rate, vol, event_grid)
    bachelier.initialization()
    n_paths = 10000
    spot = np.arange(2, 100, 4)
    mc_call = np.zeros(spot.size)
    mc_put = np.zeros(spot.size)
    for idx, s in enumerate(spot):
        paths = bachelier.paths(s, n_paths, antithetic=True)
        discounted_payoff \
            = np.maximum(paths[-1] - strike, 0) * math.exp(-rate * expiry)
        mc_call[idx] = np.sum(discounted_payoff) / n_paths
        discounted_payoff \
            = np.maximum(strike - paths[-1], 0) * math.exp(-rate * expiry)
        mc_put[idx] = np.sum(discounted_payoff) / n_paths
    # Call option
#    c = call.Call(rate, vol, event_grid, strike, expiry_idx)
    c = call.Call(rate, vol, strike, expiry_idx, event_grid)
    plt.plot(spot, c.payoff(spot), "k", label="Payoff")
    plt.plot(spot, c.price(spot, 0), "r", label="Analytical")
    plt.plot(spot, mc_call, "ob", label="Monte-Carlo")
    plt.xlabel("Stock price")
    plt.ylabel("Call option price")
    plt.show()
    # Put option
    p = put.Put(rate, vol, event_grid, strike, expiry_idx)
    plt.plot(spot, p.payoff(spot), "k", label="Payoff")
    plt.plot(spot, p.price(spot, 0), "r", label="Analytical")
    plt.plot(spot, mc_put, "ob", label="Monte-Carlo")
    plt.xlabel("Stock price")
    plt.ylabel("Put option price")
    plt.show()

    # b1 = binary.BinaryAssetCall(rate, vol, event_grid, strike, expiry_idx)
    # b2 = binary.BinaryCashCall(rate, vol, event_grid, strike, expiry_idx)
    # plt.plot(spot, c.payoff(spot), '-k')
    # plt.plot(spot, c.price(spot, time_idx), '-ob')
    # plt.plot(spot, b1.price(spot, time_idx), '-r')
    # plt.plot(spot, strike * b2.price(spot, time_idx), '-g')
    # call_price_decomposed = \
    #     b1.price(spot, time_idx) - strike * b2.price(spot, time_idx)
    # plt.plot(spot, call_price_decomposed, '-y')
    # plt.show()

    unittest.main()
