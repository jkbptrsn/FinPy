from datetime import datetime
import numpy as np
import time

from models.black_scholes import call as bs_call
from models.black_scholes import put as bs_put
from models.bachelier import call as ba_call
from models.bachelier import put as ba_put
from models.vasicek import zero_coupon_bond as va_bond
from models.vasicek import call_option as va_call
from models.vasicek import put_option as va_put

from models.cox_ingersoll_ross import zero_coupon_bond as cir_bond

from numerical_methods.finite_difference import theta
from utils import payoffs
from utils import plots

# TODO: Check analytical expression for gamma of call/put option in Vasicek model

smoothing = False
rannacher_stepping = False

show_plots = True

# model = "Black-Scholes"
# model = "Bachelier"
# model = "Vasicek"
model = "Extended Vasicek"
# model = "CIR"

# instrument = "Call"
# instrument = "Put"
instrument = "ZCBond"

bc_type = "Linearity"
# bc_type = "PDE"

solver_type = "AndersenPiterbarg"
# solver_type = "Andreasen"

# Time execution
start_time = datetime.now()

rate = 0.1
strike = 0.5  # 1.5  # 50
vol = 0.05  # 0.3
expiry = 10  # 2
kappa = 0.2
mean_rate = 0.05

t_min = 0
t_max = expiry
t_steps = 201
dt = (t_max - t_min) / (t_steps - 1)

x_min = 5
x_max = 125
x_steps = 51

if model in ("Vasicek", "Extended Vasicek"):
    x_min = -0.5  # -1.1
    x_max = 0.5   # 1.1
    x_steps = 201

if model == "CIR":
    x_min = 0.01
    x_max = 0.5
    x_steps = 201

# Reset current time
t_current = t_max

# Set up PDE solver
if solver_type == "AndersenPiterbarg":
    solver = theta.AndersenPiterbarg1D(x_min, x_max, x_steps, dt, bc_type=bc_type)
elif solver_type == "Andreasen":
    solver = theta.Andreasen1D(x_min, x_max, x_steps, dt)

if model == 'Black-Scholes':
    solver.set_drift(rate * solver.grid())
    solver.set_diffusion(vol * solver.grid())
    solver.set_rate(rate + 0 * solver.grid())
elif model == 'Bachelier':
    solver.set_drift(0 * solver.grid())
    solver.set_diffusion(vol + 0 * solver.grid())
    solver.set_rate(rate + 0 * solver.grid())
elif model == 'Vasicek':
    solver.set_drift(kappa * (mean_rate - solver.grid()))
    solver.set_diffusion(vol + 0 * solver.grid())
    solver.set_rate(solver.grid())
elif model == 'CIR':
    solver.set_drift(kappa * (mean_rate - solver.grid()))
    solver.set_diffusion(vol * np.sqrt(solver.grid()))
    solver.set_rate(solver.grid())

elif model == "Extended Vasicek":
    y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_current)) / (2 * kappa)
    solver.set_drift(y - kappa * solver.grid())
    solver.set_diffusion(vol + 0 * solver.grid())
    solver.set_rate(solver.grid())

# Terminal solution to PDE
if instrument == 'Call':
    solver.solution = payoffs.call(solver.grid(), strike)
    if model == "Vasicek" or model == "Extended Vasicek":
        solver.solution = payoffs.zero_coupon_bond(solver.grid())
elif instrument == 'Put':
    solver.solution = payoffs.put(solver.grid(), strike)
    if model == "Vasicek" or model == "Extended Vasicek":
        solver.solution = payoffs.zero_coupon_bond(solver.grid())
elif instrument == 'ZCBond':
    solver.solution = payoffs.zero_coupon_bond(solver.grid())

solver.initialization()

payoff = solver.solution.copy()

# Propagate value vector backwards in time
t1 = time.time()

# Initial step
if bc_type == "PDE":
    dt_save = solver.dt
    solver.dt = 0.00001 * dt_save
    solver.propagation()
    solver.dt = dt_save

for t in range(t_steps - 1):
    solver.propagation()
    # Update current time
    t_current -= dt

    if model == "Extended Vasicek":
        y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_current)) / (2 * kappa)
        solver.set_drift(y - kappa * solver.grid())
        solver.set_diffusion(vol + 0 * solver.grid())
        solver.set_rate(solver.grid())

        # TODO: Is this correct in terms of the time flow?
        solver.set_boundary_conditions_dt()
        solver.set_propagator()

    # European call option on ZC bond
    if t == (t_steps - 1) // 2 and (model == "Vasicek" or model == "Extended Vasicek"):
        if instrument == "Call":
            solver.solution = np.maximum(solver.solution - strike, 0)
        if instrument == "Put":
            solver.solution = np.maximum(strike - solver.solution, 0)

t2 = time.time()
print("Execution time = ", t2 - t1)

# Analytical result
instru = None
if instrument == 'Call':
    if model == 'Black-Scholes':
        instru = bs_call.Call(rate, vol, np.array([0, expiry]), strike, 1)
    elif model == "Bachelier":
        instru = ba_call.Call(vol, strike, np.array([0, expiry]), strike, 1)
    elif model == "Vasicek":
        instru = va_call.Call(kappa, mean_rate, vol, np.array([0, expiry / 2, expiry]), strike, 1, 2)
elif instrument == 'Put':
    if model == 'Black-Scholes':
        instru = bs_put.Put(rate, vol, np.array([0, expiry]), strike, 1)
    elif model == "Bachelier":
        instru = ba_put.Put(vol, strike, np.array([0, expiry]), strike, 1)
    elif model == "Vasicek":
        instru = va_put.Put(kappa, mean_rate, vol, np.array([0, expiry / 2, expiry]), strike, 1, 2)
elif instrument == 'ZCBond':
    if model == "Vasicek" or model == "Extended Vasicek":
        instru = va_bond.ZCBond(kappa, mean_rate, vol, np.array([0, expiry]), 1)
    elif model == "CIR":
        instru = cir_bond.ZCBond(kappa, mean_rate, vol, np.array([0, expiry]), 1)


plots.plot1(solver, payoff, solver.solution, instrument=instru, show=show_plots)

value = solver.solution

# Time execution
end_time = datetime.now()
print("Computation time in seconds: ", end_time - start_time)
