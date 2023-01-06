from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils.payoffs as payoffs
import models.black_scholes.call as bs_call
import models.bachelier.call as ba_call
import utils.plots as plots

import numerical_methods.finite_difference.theta as theta

n_doubles = 1

# Test convergence wrt to time and space separately

smoothing = False
rannacher_stepping = False

show_plots = True

model = "Black-Scholes"
# model = "Bachelier"

# instrument = 'Call'
instrument = 'Put'

# bc_type = "Linearity"
bc_type = "PDE"

# solver_type = "AndersenPiterbarg"
solver_type = "Andreasen"

# Time execution
start_time = datetime.now()

rate = 0.02
strike = 50
vol = 0.2
expiry = 4

t_min = 0
t_max = 4
t_steps = 501
dt = (t_max - t_min) / (t_steps - 1)

x_min = 25
x_max = 75
x_steps = 501

t_array = np.zeros(n_doubles - 1)
x_array = np.zeros(n_doubles - 1)
norm_array = np.zeros((3, n_doubles - 1))

for n in range(n_doubles):

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

    # Figure 1
    f1, ax1 = plt.subplots(3, 1)
    ax1[0].plot(solver.grid(), solver.solution, 'k')
    ax1[0].set_ylabel("Price of option")

    # Figure 2
    f2, ax2 = plt.subplots(3, 1)
    ax2[0].plot(solver.grid(), solver.solution, 'k')
    ax2[0].set_ylabel("Price of option")

    # Propagate value vector backwards in time
    for t in range(t_steps - 1):
        solver.propagation()

        # Update current time
        t_current -= dt

        # European call option on ZC bond
        if t == (t_steps - 1) // 2 and (model == "Vasicek" or model == "Extended Vasicek"):
            if instrument == "Call":
                solver.solution = np.maximum(solver.solution - strike, 0)
            if instrument == "Put":
                solver.solution = np.maximum(strike - solver.solution, 0)

    # Price function
    ax1[0].plot(solver.grid(), solver.solution, 'r')
    ax2[0].plot(solver.grid(), solver.solution, 'r')

    # Delta
    ax1[1].plot(solver.grid(), solver.delta_fd(), 'r')
    ax1[1].set_ylabel("Delta")

    # Gamma
    ax1[2].plot(solver.grid(), solver.gamma_fd(), 'r')
    ax1[2].set_ylabel("Gamma")
    ax1[2].set_xlabel("Price of underlying")

    # Theta
    ax2[1].plot(solver.grid(), solver.theta_fd(), 'r')
    ax2[1].set_ylabel("Theta")

    ax2[2].set_xlabel("Price of underlying")

    # Analytical result
    instru = None
    if instrument == 'Call':
        if model == 'Black-Scholes':
            instru = bs_call.Call(rate, vol, np.array([0, 0, 0, 0, expiry]), strike, expiry)
        elif model == "Bachelier":
            instru = ba_call.Call(vol, strike, expiry)

    plots.plot1(solver, payoff, solver.solution, instrument=instru, show=show_plots)

    value = solver.solution

#    print(solver.grid())

    # DO THIS AS THE END!!!
    # Update grid spacing in spatial dimension
    x_steps_old = x_steps
    x_steps = 2 * x_steps - 1

    # DO THIS AT THE END!!!
    # Update grid spacing in time dimension
    t_steps = 2 * t_steps - 1
    dt = (t_max - t_min) / (t_steps - 1)

# Time execution
end_time = datetime.now()
print("Computation time in seconds: ", end_time - start_time)

if show_plots:
    plt.show()

# Linear regression
res_1t = scipy.stats.linregress(t_array, norm_array[0, :])
res_2t = scipy.stats.linregress(t_array, norm_array[1, :])
res_3t = scipy.stats.linregress(t_array, norm_array[2, :])
res_1x = scipy.stats.linregress(x_array, norm_array[0, :])
res_2x = scipy.stats.linregress(x_array, norm_array[1, :])
res_3x = scipy.stats.linregress(x_array, norm_array[2, :])

print("\nLinear regression: Time dimension:")
print(res_1t.intercept, res_1t.slope)
print(res_2t.intercept, res_2t.slope)
print(res_3t.intercept, res_3t.slope)

print("\nLinear regression: Spatial dimension:")
print(res_1x.intercept, res_1x.slope)
print(res_2x.intercept, res_2x.slope)
print(res_3x.intercept, res_3x.slope)
