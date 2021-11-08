import matplotlib.pyplot as plt
import numpy as np

import models.bachelier.call as ba_call
import models.bachelier.put as ba_put
import models.black_scholes.call as bs_call
import models.black_scholes.put as bs_put
import mc_methods.standard_mc as mc

rate = 0.06
vol = 0.2
strike = 1
expiry = 1

option = ba_call.Call(vol, strike, expiry)
# option = ba_put.Put(vol, strike, expiry)
# option = bs_call.Call(rate, vol, strike, expiry)
# option = bs_put.Put(rate, vol, strike, expiry)

spot_max = 1.9
spot_min = 0.1
n_grid = 51
spot_grid = (spot_max - spot_min) \
            * np.array(range(n_grid)) / (n_grid - 1) + spot_min

n_paths = 200000
mc_gen = mc.MonteCarlo(n_paths, option)

# show = 'price'
show = 'greek'

if show == 'price':
    mean, std = mc_gen.price(spot_grid, 0, antithetic=True)
    plt.plot(spot_grid, option.payoff(spot_grid), 'k')
    plt.plot(spot_grid, option.price(spot_grid, 0), 'b')
    plt.errorbar(spot_grid, mean, fmt='ro', ecolor='r', yerr=std,
                 markersize=3, capsize=2)
    plt.xlabel('Spot')
    plt.ylabel('Option price')
    plt.show()
elif show == 'greek':
    mean, std = mc_gen.greek(spot_grid, 0, 'delta', 'path-wise', False)
    plt.plot(spot_grid, option.payoff(spot_grid), 'k')
    plt.plot(spot_grid, option.delta(spot_grid, 0), 'b')
    plt.errorbar(spot_grid, mean, fmt='ro', ecolor='r', yerr=std,
                 markersize=3, capsize=2)
    plt.xlabel('Spot')
    plt.ylabel('Option price / Delta')
    plt.show()
