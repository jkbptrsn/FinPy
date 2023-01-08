from datetime import datetime
import numpy as np

from models.black_scholes import call as bs_call
from models.bachelier import call as ba_call
from numerical_methods.finite_difference import theta
from utils import payoffs
from utils import plots

n_doubles = 1

# Test convergence wrt to time and space separately

smoothing = False
rannacher_stepping = False

show_plots = True

model = "Black-Scholes"
# model = "Bachelier"

instrument = 'Call'
# instrument = 'Put'

bc_type = "Linearity"
# bc_type = "PDE"

# solver_type = "AndersenPiterbarg"
solver_type = "Andreasen"

# Time execution
start_time = datetime.now()

rate = 0.1
strike = 50
vol = 0.3
expiry = 4

t_min = 0
t_max = expiry
t_steps = 101
dt = (t_max - t_min) / (t_steps - 1)

x_min = 5
x_max = 145
x_steps = 51

t_array = np.zeros(n_doubles - 1)
x_array = np.zeros(n_doubles - 1)
norm_array = np.zeros((3, n_doubles - 1))

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
    solver.set_rate(0 * solver.grid())

# Terminal solution to PDE
if instrument == 'Call':
    solver.solution = payoffs.call(solver.grid(), strike)
elif instrument == 'Put':
    solver.solution = payoffs.put(solver.grid(), strike)

solver.initialization()

payoff = solver.solution.copy()

# Propagate value vector backwards in time
for t in range(t_steps - 1):
    solver.propagation()
    # Update current time
    t_current -= dt

# Analytical result
instru = None
if instrument == 'Call':
    if model == 'Black-Scholes':
        instru = bs_call.Call(rate, vol, np.array([0, expiry]), strike, 1)
    elif model == "Bachelier":
        instru = ba_call.Call(vol, strike, expiry)

plots.plot1(solver, payoff, solver.solution, instrument=instru, show=show_plots)

value = solver.solution

# Time execution
end_time = datetime.now()
print("Computation time in seconds: ", end_time - start_time)
