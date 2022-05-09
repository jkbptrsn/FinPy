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

n_doubles = 3

# Test convergence wrt to time and space separately

smoothing = False

show_plots = False

rannacher_stepping = False

# model = "Black-Scholes"
# model = "Bachelier"
model = "Vasicek"
# model = "Extended Vasicek"

# instrument = 'Call'
# instrument = 'Put'
instrument = 'ZCBond'

# Time execution
start_time = datetime.now()

rate = 0.0
strike = 1 # 50
vol = 0.05 # 0.2
expiry = 2
kappa = 0.1 # 1.0 # 0.1
theta_factor = 0.0

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
x_steps = 101 + 2 # 101

t_array = np.zeros(n_doubles - 1)
x_array = np.zeros(n_doubles - 1)
norm_array = np.zeros((3, n_doubles - 1))

for n in range(n_doubles):

    # Update grid spacing in spatial dimension
    x_steps_old = x_steps
#    x_steps = 2 * x_steps - 1


    x_steps = 2 * (x_steps - 2) - 1 + 2
    print(x_steps)


    # Update grid spacing in time dimension
    t_steps = 2 * t_steps - 1
    dt = (t_max - t_min) / (t_steps - 1)

    # Reset current time
    t_current = t_max

    # Set up PDE solver
    solver = theta.Solver(x_min, x_max, x_steps)
    solver.initialization()
    if model == 'Black-Scholes':
        solver.set_up_drift_vec(rate * solver.grid())
        solver.set_up_diffusion_vec(vol * solver.grid())
        solver.set_up_rate_vec(rate + 0 * solver.grid())
    if model == 'Bachelier':
        solver.set_up_drift_vec(0 * solver.grid())
        solver.set_up_diffusion_vec(vol + 0 * solver.grid())
        solver.set_up_rate_vec(0 * solver.grid())
    elif model == 'Vasicek':
        solver.set_up_drift_vec(kappa * (theta_factor - solver.grid()))
        solver.set_up_rate_vec(solver.grid())
        solver.set_up_diffusion_vec(vol + 0 * solver.grid())
    elif model == "Extended Vasicek":
        # Add time-dependent volatility and non-zero forward rate
        y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_current)) / (2 * kappa)
        solver.set_up_drift_vec(y - kappa * solver.grid())

#        nu = rate + y / kappa
#        solver.set_up_drift_vec(kappa * (nu - solver.grid()))

        solver.set_up_rate_vec(solver.grid())
        solver.set_up_diffusion_vec(vol + 0 * solver.grid())

    solver.set_up_propagator(dt)

    # Terminal solution to PDE
    if instrument == 'Call':
        value = payoffs.call(solver.grid(), strike)
        if model == "Vasicek" or model == "Extended Vasicek":
            value = 1 + 0 * solver.grid()
    elif instrument == 'Put':
        value = payoffs.put(solver.grid(), strike)
        if model == "Vasicek" or model == "Extended Vasicek":
            value = 1 + 0 * solver.grid()
    elif instrument == 'ZCBond':
        value = 1 + 0 * solver.grid()

    if smoothing:
        if instrument == "Call" and not (model == "Vasicek" or model == "Extended Vasicek"):
            grid = solver.grid()
            dx = grid[1] - grid[0]
            index = np.where(grid < strike)[0][-1]
            if (strike - grid[index]) < dx / 2:
                dist = dx / 2 - (strike - grid[index])
                integral = dist ** 2 / 2
                value[index] = integral / dx
            else:
                dist = dx / 2 + (grid[index + 1] - strike)
                integral = dist ** 2 / 2
                value[index + 1] = integral / dx
                if grid[index + 1] == strike:
                    value[index + 1] = 0

    payoff = value

    # Figure 1
    f1, ax1 = plt.subplots(3, 1)
    ax1[0].plot(solver.grid(), value, 'k')
    ax1[0].set_ylabel("Price of option")

    # Figure 2
    f2, ax2 = plt.subplots(3, 1)
    ax2[0].plot(solver.grid(), value, 'k')
    ax2[0].set_ylabel("Price of option")

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
                solver.set_up_drift_vec(kappa * (theta_factor - solver.grid()))
            elif model == "Extended Vasicek":
                y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_temp)) / (2 * kappa)
#                print(t_current, y)
                solver.set_up_drift_vec(y - kappa * solver.grid())

#                nu = rate + y / kappa
#                solver.set_up_drift_vec(kappa * (nu - solver.grid()))

            solver.set_up_rate_vec(solver.grid())
            solver.set_up_diffusion_vec(vol + 0 * solver.grid())
            solver.set_up_propagator(dt)

        if rannacher_stepping and t < 2:
            solver.theta = 1.0
            value = solver.propagation(dt, value)
            solver.theta = 0.5
        else:
            value = solver.propagation(dt, value)

        # Update current time
        t_current -= dt

        # European call option on ZC bond
        if t == (t_steps - 1) // 2 and (model == "Vasicek" or model == "Extended Vasicek"):
            if instrument == "Call":
                value = np.maximum(value - strike, 0)
            if instrument == "Put":
                value = np.maximum(strike - value, 0)

    # Price function
    ax1[0].plot(solver.grid(), value, 'r')
    ax2[0].plot(solver.grid(), value, 'r')

    # Delta
    ax1[1].plot(solver.grid(), solver.fd_delta(solver.grid(), value), 'r')
    ax1[1].set_ylabel("Delta")

    # Gamma
    ax1[2].plot(solver.grid(), solver.fd_gamma(solver.grid(), value), 'r')
    ax1[2].set_ylabel("Gamma")
    ax1[2].set_xlabel("Price of underlying")

    # Theta
    ax2[1].plot(solver.grid(), solver.fd_theta(dt, value), 'r')
    ax2[1].set_ylabel("Theta")

    plots.plot1(solver, payoff, value, dt, show=show_plots)

    ax2[2].set_xlabel("Price of underlying")

    # Analytical result
    if instrument == 'Call':
        if model == 'Black-Scholes':
            call = bs_call.Call(rate, vol, strike, expiry)
        elif model == "Bachelier":
            call = ba_call.Call(vol, strike, expiry)
        elif model == "Vasicek":
            call = va_call.Call(kappa, theta_factor, vol, strike, expiry / 2, expiry)
        elif model == "Extended Vasicek":
            # UPDATE !!!
            call = va_call.Call(kappa, theta_factor, vol, strike, expiry / 2, expiry)
        ax1[0].plot(solver.grid(), call.price(solver.grid(), 0), 'ob', markersize=3)
        ax1[1].plot(solver.grid(), call.delta(solver.grid(), 0), 'ob', markersize=3)
        ax2[0].plot(solver.grid(), call.price(solver.grid(), 0), 'ob', markersize=3)
        if model == "Black-Scholes" or model == "Bachelier":
            ax1[2].plot(solver.grid(), call.gamma(solver.grid(), 0), 'ob', markersize=3)
            ax2[1].plot(solver.grid(), call.theta(solver.grid(), 0), 'ob', markersize=3)
    elif instrument == 'Put':
        if model == 'Black-Scholes':
            put = bs_put.Put(rate, vol, strike, expiry)
        elif model == "Bachelier":
            put = ba_put.Put(vol, strike, expiry)
        elif model == "Vasicek":
            put = va_put.Put(kappa, theta_factor, vol, strike, expiry / 2, expiry)
        elif model == "Extended Vasicek":
            # UPDATE !!!
            put = va_put.Put(kappa, theta_factor, vol, strike, expiry / 2, expiry)
        ax1[0].plot(solver.grid(), put.price(solver.grid(), 0), 'ob', markersize=3)
        ax1[1].plot(solver.grid(), put.delta(solver.grid(), 0), 'ob', markersize=3)
        ax2[0].plot(solver.grid(), put.price(solver.grid(), 0), 'ob', markersize=3)
        if model == "Vasicek" or model == "Extended Vasicek":
            ax1[2].plot(solver.grid(), put.gamma(solver.grid(), 0), 'ob', markersize=3)
            ax2[1].plot(solver.grid(), put.theta(solver.grid(), 0), 'ob', markersize=3)
    elif instrument == 'ZCBond':
        zcbond = va_bond.ZCBond(kappa, theta_factor, vol, strike, expiry)
        ax1[0].plot(solver.grid(), zcbond.price(solver.grid(), 0), 'ob', markersize=3)
        ax1[1].plot(solver.grid(), zcbond.delta(solver.grid(), 0), 'ob', markersize=3)
        ax1[2].plot(solver.grid(), zcbond.gamma(solver.grid(), 0), 'ob', markersize=3)


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

        norm_center = abs_diff[(x_steps_old - 1) // 2]
        norm_max = np.amax(abs_diff)
        norm_l2 = np.sqrt(np.sum(2 * solver.dx * np.square(abs_diff)))

        print(t_steps, x_steps_old, norm_center, norm_max, norm_l2)

        # Data used for linear regression
        t_array[n - 1] = np.log(dt)
        x_array[n - 1] = np.log(solver.dx)
        norm_array[0, n - 1] = np.log(norm_center)
        norm_array[1, n - 1] = np.log(norm_max)
        norm_array[2, n - 1] = np.log(norm_l2)
    # Save value vector
    value_old = value

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
