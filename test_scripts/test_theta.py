from datetime import datetime
import numpy as np

from models.bachelier import call as ba_call
from models.black_scholes import call as bs_call
from models.black_scholes import put as bs_put
from models.cox_ingersoll_ross import zero_coupon_bond as cir_bond
from models.vasicek import zero_coupon_bond as va_bond
from utils import plots


# Time execution
start_time = datetime.now()

rate = 0.1
strike = 50
vol = 0.3
expiry = 2
kappa = 0.5
mean_rate = 0.1

x_min = 5
x_max = 125
x_steps = 51

t_min = 0
t_max = expiry
t_steps = 201
dt = (t_max - t_min) / (t_steps - 1)

expiry_idx = t_steps - 1
event_grid = dt * np.arange(t_steps) - t_min
maturity_idx = t_steps - 1

# model_name = "Black-Scholes"
# model_name = "Bachelier"
model_name = "Vasicek"
# model_name = "Extended Vasicek"
# model_name = "CIR"

# instrument = "Call"
# instrument = "Put"
instrument = "ZCBond"

if model_name in ("Vasicek", "Extended Vasicek"):
    strike = 0.5
    vol = 0.05
    expiry = 10
    x_min = -0.5  # -1.1
    x_max = 0.5   # 1.1
    x_steps = 201

if model_name == "CIR":
    strike = 0.5
    vol = 0.1
    expiry = 10
    x_min = 0.01
    x_max = 0.5
    x_steps = 201

dx = (x_max - x_min) / (x_steps - 1)
x_grid = dx * np.arange(x_steps) + x_min

if model_name == "Black-Scholes":
    if instrument == "Call":
        instru = bs_call.Call(rate, vol, strike, expiry_idx, event_grid)
    elif instrument == "Put":
        instru = bs_put.Put(rate, vol, strike, expiry_idx, event_grid)
elif model_name == "Bachelier":
    if instrument == "Call":
        instru = ba_call.Call(rate, vol, strike, expiry_idx, event_grid)
elif model_name == "Vasicek":
    if instrument == "ZCBond":
        instru = va_bond.ZCBond(kappa, mean_rate, vol, maturity_idx, event_grid)
elif model_name == "CIR":
    if instrument == "ZCBond":
        instru = cir_bond.ZCBond(kappa, mean_rate, vol, event_grid, maturity_idx)

# instru.fd_setup(x_min, x_max, x_steps)
instru.fd_setup(x_grid, equidistant=True)

payoff = instru.fd.solution.copy()
instru.fd_solve()
plots.plot_price_and_greeks(instru, payoff, instru.fd.solution, show=True)
