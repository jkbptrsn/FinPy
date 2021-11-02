import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

import models.payoffs as payoffs
import models.black_scholes.call as bs_call

rate = 0.02
strike = 1.0
vol = 0.2
volsq = vol ** 2
expiry = 1

t_min = 0
t_max = 1
t_steps = 5001
dt = (t_max - t_min) / (t_steps - 1)
t_grid = dt * np.array(range(t_steps, 0, -1))
dt *= -1

x_min = 0
x_max = 4
x_steps = 101
dx = (x_max - x_min) / (x_steps - 1)
x_grid = dx * np.array(range(x_steps)) + x_min

p_grid = np.zeros((t_steps, x_steps))
# Final-time condition
p_grid[0, :] = payoffs.call(x_grid, strike)
# Boundary condition
p_grid[:, -1] = x_grid[-1] - strike * np.exp(-rate * (expiry - t_grid))

alpha = volsq * x_grid ** 2 / (4 * dx ** 2)
beta = rate * x_grid / (4 * dx)

a_factor = alpha * dt + beta * dt
b_factor = alpha * dt - beta * dt
c_factor = 1 - 2 * alpha * dt - rate * dt / 2
d_factor = 1 + 2 * alpha * dt + rate * dt / 2

tridiag = np.ndarray((3, x_steps))
for n in range(x_steps - 1):
    tridiag[0, n + 1] = a_factor[n]
tridiag[1, :] = c_factor
for n in range(x_steps - 1):
    tridiag[2, n] = b_factor[n + 1]

for t in range(1, t_steps):

    b_vector = p_grid[t - 1, :]

    b_vector[1:-1] = \
        - a_factor[1:-1] * b_vector[2:] \
        + d_factor[1:-1] * b_vector[1:-1] \
        - b_factor[1:-1] * b_vector[:-2]

    p_grid[t, :] = solve_banded((1, 1), tridiag, b_vector)

    # Maintain boundary
    p_grid[t, -1] = x_grid[-1] - strike * np.exp(-rate * (expiry - t_grid[t]))

plt.plot(x_grid, p_grid[0, :], 'k')
plt.plot(x_grid, p_grid[-1, :], 'r')
call = bs_call.Call(rate, vol, strike, expiry)
plt.plot(x_grid, call.price(x_grid, 0), '.b')
#plt.xlim((0.5, 1.5))
#plt.ylim((-0.1, 0.6))
plt.show()




