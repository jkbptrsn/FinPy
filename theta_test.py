import math
import matplotlib.pyplot as plt
import numpy as np

import models.payoffs as payoffs
import models.black_scholes.call as bs_call
import models.black_scholes.put as bs_put
import fd_methods.theta as theta


rate = 0.06
strike = 40.0
vol = 0.2
expiry = 1

t_min = 0
t_max = 1
t_steps = 2001
dt = (t_max - t_min) / (t_steps - 1)

x_min = 0.0
x_max = 160.0
x_steps = 3201

PDEsolver = theta.Solver(x_min, x_max, x_steps)

# Differential operator in Black-Scholes PDE
d_operator = - rate * PDEsolver.identity() + rate * PDEsolver.x_ddx() + 0.5 * vol ** 2 * PDEsolver.x2_d2dx2()
option = 'put'
exercise_type = 'American'
# Final conditions
if option == 'call':
    v_vector = payoffs.call(PDEsolver.x_grid(), strike)
elif option == 'put':
    v_vector = payoffs.put(PDEsolver.x_grid(), strike)
plt.plot(PDEsolver.x_grid(), v_vector, 'k')
# Analytic result
if option == 'call':
    call = bs_call.Call(rate, vol, strike, expiry)
    plt.plot(PDEsolver.x_grid(), call.price(PDEsolver.x_grid(), 0), 'b')
elif option == 'put':
    put = bs_put.Put(rate, vol, strike, expiry)
    plt.plot(PDEsolver.x_grid(), put.price(PDEsolver.x_grid(), 0), 'b')

for t in range(t_steps):
    v_vector = PDEsolver.propagation(dt, d_operator, v_vector)
    if t % ((t_steps - 1) // 50) == 0:
        if exercise_type == 'American':
            if option == 'call':
                v_vector = np.maximum(v_vector, PDEsolver.x_grid() - strike)
            elif option == 'put':
                v_vector = np.maximum(v_vector, strike - PDEsolver.x_grid())

plt.plot(PDEsolver.x_grid(), v_vector, '.r')
#plt.xlim((0.2, 1.4))
#plt.ylim((-0.1, 0.4))
plt.show()

s_points = [36, 38, 40, 42, 44]
n_mc_paths = 100000
for x, v1, v2 in zip(PDEsolver.x_grid(), v_vector, put.price(PDEsolver.x_grid(), 0)):
    for s in s_points:
        if math.fabs(x - s) < 1e-5:
            if option == 'call':
                s_mc = call.path(s, expiry, n_mc_paths, antithetic=True)
                s_mc = call.payoff(s_mc)
            elif option == 'put':
                s_mc = put.path(s, expiry, n_mc_paths, antithetic=True)
                s_mc = put.payoff(s_mc)
            s_mc = math.exp(-rate * expiry) * sum(s_mc) / len(s_mc)
            print(x, v1, v2, math.fabs(v1 - v2), s_mc)
