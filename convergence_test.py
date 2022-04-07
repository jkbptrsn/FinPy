from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils.payoffs as payoffs
import models.black_scholes.call as bs_call
import models.black_scholes.put as bs_put
import models.bachelier.call as ba_call
import fd_methods.theta as theta

n_doubles = 3

# model = 'Black-Scholes'
model = 'Vasicek'

# option_type = 'Call'
# option_type = 'Put'
option_type = 'ZCBond'

# Time execution
start_time = datetime.now()

rate = 0.0
strike = 50.0
vol = 0.05 # 0.2
expiry = 2
kappa = 1.0 # 0.1

t_min = 0
t_max = 1
t_steps = 2001
dt = (t_max - t_min) / (t_steps - 1)

print("STD: ", np.sqrt(vol ** 2 * (t_max - t_min)))

x_min = -5 # 5.0
x_max = 5 # 95.0
x_steps = 2001

t_array = np.zeros(n_doubles - 1)
x_array = np.zeros(n_doubles - 1)
norm_array = np.zeros((3, n_doubles - 1))

for n in range(n_doubles):
    # Update grid spacing in spatial dimension
    x_steps_old = x_steps
    x_steps = 2 * x_steps - 1
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
    elif model == 'Vasicek':
        # Add time-dependent volatility and non-zero forward rate
        # Integrate up to time t_current
        y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_current)) / (2 * kappa)
        print(y, np.amax(np.abs(y - kappa * solver.grid()) * solver.dx), vol ** 2)


#        y = 0.01

        solver.set_up_drift_vec(y - kappa * solver.grid())
#        solver.set_up_drift_vec(kappa * (0 - solver.grid()))

        solver.set_up_diffusion_vec(vol + 0 * solver.grid())
        solver.set_up_rate_vec(rate + 0 * solver.grid())
#        solver.set_up_rate_vec(rate + 0 * solver.grid())
    solver.set_up_propagator()

    # Terminal solution to PDE
    if option_type == 'Call':
        value = payoffs.call(solver.grid(), strike)
    elif option_type == 'Put':
        value = payoffs.put(solver.grid(), strike)

    elif option_type == 'ZCBond':
        value = payoffs.call(solver.grid(), 0)
#        value = 1 + 0 * solver.grid()

    plt.plot(solver.grid(), value, 'k')

    # Propagate value vector backwards in time
    for t in range(t_steps - 1):

        # Update current time
        t_current -= dt

        if model == 'Vasicek':
            # Add time-dependent volatility and non-zero forward rate
            # Integrate up to time t_current
            y = vol ** 2 * (1 - np.exp(- 2 * kappa * t_current)) / (2 * kappa)

#            y = 0.1

            solver.set_up_drift_vec(y - kappa * solver.grid())
#            solver.set_up_drift_vec(kappa * ( 0 - solver.grid()))

            solver.set_up_diffusion_vec(vol + 0 * solver.grid())
#            solver.set_up_rate_vec(rate + 0 * solver.grid())
            solver.set_up_rate_vec(rate + 0 * solver.grid())
            solver.set_up_propagator()

        value = solver.propagation(dt, value)
    plt.plot(solver.grid(), value, 'r')

    # Analytical result
    if option_type == 'Call':
        call = bs_call.Call(rate, vol, strike, expiry)
        plt.plot(solver.grid(), call.price(solver.grid(), 0), 'ob', markersize=3)
    elif option_type == 'Put':
        put = bs_put.Put(rate, vol, strike, expiry)
        plt.plot(solver.grid(), put.price(solver.grid(), 0), 'ob', markersize=3)
    elif option_type == 'ZCBond':
        call = ba_call.Call(vol, 0, expiry)
        plt.plot(solver.grid(), call.price(solver.grid(), 0), 'ob', markersize=3)
        # zcbond = np.exp(- solver.grid() * (t_max - 0))
        # plt.plot(solver.grid(), zcbond, 'ob', markersize=3)

    if n > 0:
        abs_diff = np.abs(value_old - value[::2])
        norm_center = abs_diff[(x_steps_old - 1) // 2]
        norm_max = np.amax(abs_diff)
        norm_l2 = np.sqrt(np.sum(2 * solver.dx * np.square(abs_diff)))
        # print(t_steps, x_steps_old, norm_center, norm_max, norm_l2)

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
