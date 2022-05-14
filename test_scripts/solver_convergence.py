from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils.payoffs as payoffs
import models.black_scholes.call as bs_call
import models.black_scholes.put as bs_put
import models.bachelier.call as ba_call
import models.bachelier.put as ba_put
import models.vasicek.zcbond as va_bond
import models.vasicek.call as va_call
import models.vasicek.put as va_put
import utils.plots as plots

import numerical_methods.finite_difference.theta as theta

# Fix BACHELIER CLASS

n_doubles = 1

# Test convergence wrt to time and space separately

smoothing = False

show_plots = True

rannacher_stepping = False

# model = "Black-Scholes"
# model = "Bachelier"
model = "Vasicek"
# model = "Extended Vasicek"

# instrument = 'Call'
# instrument = 'Put'
instrument = 'ZCBond'

bc_type = "Linearity"
# bc_type = "PDE"

solver_type = "AndersenPiterbarg"
# solver_type = "Andreasen"

# Time execution
start_time = datetime.now()

rate = 0.0
strike = 0.2 # 50
vol = 0.05 # 0.2
expiry = 2
kappa = 0.1 # 1.0 # 0.1
theta_factor = 0

t_min = 0
t_max = 2
t_steps = 101
dt = (t_max - t_min) / (t_steps - 1)

sigma_grid = np.sqrt(vol ** 2 * (t_max - t_min))
print("STD: ", sigma_grid)

sigma_grid_new = np.sqrt(vol ** 2 * (1 - np.exp(-2 * kappa * (t_max - t_min))) / (2 * kappa))
print("STD new: ", sigma_grid_new)

x_min = - 5 * sigma_grid # 25
x_max = 5 * sigma_grid # 75
x_steps = 101

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
    if model == 'Bachelier':
        solver.set_drift(0 * solver.grid())
        solver.set_diffusion(vol + 0 * solver.grid())
        solver.set_rate(0 * solver.grid())
    elif model == 'Vasicek':
        solver.set_drift(kappa * (theta_factor - solver.grid()))
        solver.set_diffusion(vol + 0 * solver.grid())
        solver.set_rate(solver.grid())
    elif model == "Extended Vasicek":
        # Add time-dependent volatility and non-zero forward rate
        y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_current)) / (2 * kappa)
        solver.set_drift(y - kappa * solver.grid())

#        nu = rate + y / kappa
#        solver.set_up_drift_vec(kappa * (nu - solver.grid()))

        solver.set_diffusion(vol + 0 * solver.grid())
        solver.set_rate(solver.grid())

    # Terminal solution to PDE
    if instrument == 'Call':
        solver.solution = payoffs.call(solver.grid(), strike)
        if model == "Vasicek" or model == "Extended Vasicek":
            solver.solution = 1 + 0 * solver.grid()
    elif instrument == 'Put':
        solver.solution = payoffs.put(solver.grid(), strike)
        if model == "Vasicek" or model == "Extended Vasicek":
            solver.solution = 1 + 0 * solver.grid()
    elif instrument == 'ZCBond':
        solver.solution = payoffs.zero_coupon_bond(solver.grid())

    solver.initialization()

    payoff = solver.solution.copy()

    # Figure 1
#    f1, ax1 = plt.subplots(3, 1)
#    ax1[0].plot(solver.grid(), solver.solution, 'k')
#    ax1[0].set_ylabel("Price of option")

    # Figure 2
#    f2, ax2 = plt.subplots(3, 1)
#    ax2[0].plot(solver.grid(), solver.solution, 'k')
#    ax2[0].set_ylabel("Price of option")

    # Propagate value vector backwards in time
    for t in range(t_steps - 1):

        if model == 'Vasicek' or model == "Extended Vasicek":

            if n == 0:
                t_temp = t_current
            elif n == 1 and t % 2 == 0:
                t_temp = t_current
            elif n == 2 and t % 4 == 0:
                t_temp = t_current
            elif n == 3 and t % 8 == 0:
                t_temp = t_current
            elif n == 4 and t % 16 == 0:
                t_temp = t_current

            if model == "Vasicek":
                solver.set_drift(kappa * (theta_factor - solver.grid()))
            elif model == "Extended Vasicek":
                y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_temp)) / (2 * kappa)
#                print(t_current, y)
                solver.set_drift(y - kappa * solver.grid())

#                nu = rate + y / kappa
#                solver.set_up_drift_vec(kappa * (nu - solver.grid()))

            solver.set_diffusion(vol + 0 * solver.grid())
            solver.set_rate(solver.grid())

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
#    ax1[0].plot(solver.grid(), solver.solution, 'r')
#    ax2[0].plot(solver.grid(), solver.solution, 'r')

    # Delta
#    ax1[1].plot(solver.grid(), solver.delta_fd(), 'r')
#    ax1[1].set_ylabel("Delta")

    # Gamma
#    ax1[2].plot(solver.grid(), solver.gamma_fd(), 'r')
#    ax1[2].set_ylabel("Gamma")
#    ax1[2].set_xlabel("Price of underlying")

    # Theta
#    ax2[1].plot(solver.grid(), solver.theta_fd(), 'r')
#    ax2[1].set_ylabel("Theta")

#    ax2[2].set_xlabel("Price of underlying")

    # Analytical result
    instru = None
    if instrument == 'Call':
        if model == 'Black-Scholes':
            instru = bs_call.Call(rate, vol, strike, expiry)
        elif model == "Bachelier":
            instru = ba_call.Call(vol, strike, expiry)
        elif model == "Vasicek":
            instru = va_call.Call(kappa, theta_factor, vol, strike, expiry / 2, expiry)
        elif model == "Extended Vasicek":
            # UPDATE !!!
            instru = va_call.Call(kappa, theta_factor, vol, strike, expiry / 2, expiry)
#        ax1[0].plot(solver.grid(), instru.price(solver.grid(), 0), 'ob', markersize=3)
#        ax1[1].plot(solver.grid(), instru.delta(solver.grid(), 0), 'ob', markersize=3)
#        ax2[0].plot(solver.grid(), instru.price(solver.grid(), 0), 'ob', markersize=3)
#        if model == "Black-Scholes" or model == "Bachelier":
#            ax1[2].plot(solver.grid(), instru.gamma(solver.grid(), 0), 'ob', markersize=3)
#            ax2[1].plot(solver.grid(), instru.theta(solver.grid(), 0), 'ob', markersize=3)
    elif instrument == 'Put':
        if model == 'Black-Scholes':
            instru = bs_put.Put(rate, vol, strike, expiry)
        elif model == "Bachelier":
            instru = ba_put.Put(rate, vol, strike, expiry)
        elif model == "Vasicek":
            instru = va_put.Put(kappa, theta_factor, vol, strike, expiry / 2, expiry)
        elif model == "Extended Vasicek":
            # UPDATE !!!
            instru = va_put.Put(kappa, theta_factor, vol, strike, expiry / 2, expiry)
#        ax1[0].plot(solver.grid(), instru.price(solver.grid(), 0), 'ob', markersize=3)
#        ax1[1].plot(solver.grid(), instru.delta(solver.grid(), 0), 'ob', markersize=3)
#        ax2[0].plot(solver.grid(), instru.price(solver.grid(), 0), 'ob', markersize=3)
#        if model == "Vasicek" or model == "Extended Vasicek":
#            ax1[2].plot(solver.grid(), instru.gamma(solver.grid(), 0), 'ob', markersize=3)
#            ax2[1].plot(solver.grid(), instru.theta(solver.grid(), 0), 'ob', markersize=3)
    elif instrument == 'ZCBond':
        instru = va_bond.ZCBond(kappa, theta_factor, vol, expiry)
#        ax1[0].plot(solver.grid(), instru.price(solver.grid(), 0), 'ob', markersize=3)
#        ax1[1].plot(solver.grid(), instru.delta(solver.grid(), 0), 'ob', markersize=3)
#        ax1[2].plot(solver.grid(), instru.gamma(solver.grid(), 0), 'ob', markersize=3)

    plots.plot1(solver, payoff, solver.solution, instrument=instru, show=show_plots)

    value = solver.solution

#    print(solver.grid())

    if n > 0:

#        abs_diff = np.abs(value_old - value[::2])

        abs_diff = np.abs(value_old[1:-1] - value[1:-1][::2])
#        abs_diff = np.abs(value_old[1:-1] - value[2:-2][::2])

#        abs_diff = np.abs(value_old - value)
#        if model == "Vasicek" or model == "Extended Vasicek":
#            if instrument == "Call":
#                call = va_call.Call(kappa, theta_factor, vol, strike, expiry / 2, expiry)
#                abs_diff = np.abs(call.price(solver.grid(), 0) - value)
#            if instrument == "ZCBond":
#                zcbond = va_bond.ZCBond(kappa, theta_factor, vol, strike, expiry)
#                abs_diff = np.abs(zcbond.price(solver.grid(), 0) - value)

#        print(solver.grid()[::2][(x_steps_old - 1) // 2])

        norm_center = abs_diff[(x_steps_old - 1) // 2]
#        norm_center = abs_diff[(x_steps_old - 1) // 2 - 1]

        norm_max = np.amax(abs_diff)
        # Old step size is 2 * self.dx
        norm_l2 = np.sqrt(np.sum((2 * solver.dx) * np.square(abs_diff)))

        print(t_steps, x_steps_old, norm_center, norm_max, norm_l2)

        # Data used for linear regression
        t_array[n - 1] = np.log(dt)
        x_array[n - 1] = np.log(solver.dx)
        norm_array[0, n - 1] = np.log(norm_center)
        norm_array[1, n - 1] = np.log(norm_max)
        norm_array[2, n - 1] = np.log(norm_l2)

    # Save value vector
    value_old = value

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
