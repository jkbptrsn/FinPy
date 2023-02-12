from datetime import datetime
import numpy as np
import time

from models.black_scholes import call as bs_call
from numerical_methods.finite_difference import theta
from utils import payoffs
from utils import plots


# Time execution
start_time = datetime.now()

rate = 0.1
strike = 50
vol = 0.3
expiry = 2
kappa = 0.2
mean_rate = 0.05

x_min = 5
x_max = 125
x_steps = 51

t_min = 0
t_max = expiry
t_steps = 201
dt = (t_max - t_min) / (t_steps - 1)

expiry_idx = t_steps - 1
event_grid = dt * np.arange(t_steps) - t_min

call = bs_call.CallNew(rate, vol, strike, expiry_idx, event_grid)
call.fd_setup(x_min, x_max, x_steps)

payoff = call.fd.solution.copy()

call.fd_solve()

instru = bs_call.Call(rate, vol, np.array([0, expiry]), strike, 1)

plots.plot_price_and_greeks(call.fd, payoff, call.fd.solution,
                            instrument=instru, show=True)
