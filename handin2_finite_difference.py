import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2

import utils.payoffs as payoffs
import models.black_scholes.call as bs_call
import models.black_scholes.put as bs_put
import fd_methods.theta as theta


rate = 0 #0.06
strike = 40.0
vol = 0.4
expiry = 2

t_min = 0
t_max = 2
t_steps = 8001
dt = (t_max - t_min) / (t_steps - 1)

x_min = 0.0
x_max = 120.0
x_steps = 3201 #3201

PDEsolver1 = theta.Solver(x_min, x_max, x_steps)
PDEsolver2 = theta.Solver(x_min, x_max, x_steps)
PDEsolver3 = theta.Solver(x_min, x_max, (x_steps - 1) // 2 + 1)
PDEsolver4 = theta.Solver(x_min, x_max, x_steps)
PDEsolver5 = theta.Solver(x_min, x_max, (x_steps - 1) // 4 + 1)

PDEsolver6 = theta.Solver(x_min, x_max, x_steps)
PDEsolver7 = theta.Solver(x_min, x_max, x_steps)

# Differential operator in Black-Scholes PDE
d_operator1 = - rate * PDEsolver1.identity() + rate * PDEsolver1.x_ddx() + 0.5 * vol ** 2 * PDEsolver1.x2_d2dx2()
d_operator2 = - rate * PDEsolver2.identity() + rate * PDEsolver2.x_ddx() + 0.5 * vol ** 2 * PDEsolver2.x2_d2dx2()
d_operator3 = - rate * PDEsolver3.identity() + rate * PDEsolver3.x_ddx() + 0.5 * vol ** 2 * PDEsolver3.x2_d2dx2()
d_operator4 = - rate * PDEsolver4.identity() + rate * PDEsolver4.x_ddx() + 0.5 * vol ** 2 * PDEsolver4.x2_d2dx2()
d_operator5 = - rate * PDEsolver5.identity() + rate * PDEsolver5.x_ddx() + 0.5 * vol ** 2 * PDEsolver5.x2_d2dx2()

# Dupire, Black-Scholes
d_operator6 = \
    - rate * PDEsolver6.identity() + rate * PDEsolver6.x_ddx() \
    - 0.5 * vol ** 2 * PDEsolver6.x2_d2dx2()
# Dupire, CEV
beta = -0.5
d_operator7 = \
    - rate * PDEsolver7.identity() + rate * PDEsolver7.x_ddx() \
    - 0.5 * vol ** 2 * PDEsolver7.vector_d2dx2(PDEsolver7.x_grid() ** (2 * beta + 2))

option = 'call'
# option = 'put'

exercise_type = 'European'
# exercise_type = 'American'

# Final conditions
if option == 'call':
    v_vector1 = payoffs.call(PDEsolver1.x_grid(), strike)
    v_vector2 = payoffs.call(PDEsolver2.x_grid(), strike)
    v_vector3 = payoffs.call(PDEsolver3.x_grid(), strike)
    v_vector4 = payoffs.call(PDEsolver4.x_grid(), strike)
    v_vector5 = payoffs.call(PDEsolver5.x_grid(), strike)

    v_vector6 = payoffs.put(PDEsolver6.x_grid(), strike)
    v_vector7 = payoffs.put(PDEsolver7.x_grid(), strike)

elif option == 'put':
    v_vector1 = payoffs.put(PDEsolver1.x_grid(), strike)
    v_vector2 = payoffs.put(PDEsolver2.x_grid(), strike)
    v_vector3 = payoffs.put(PDEsolver3.x_grid(), strike)
    v_vector4 = payoffs.put(PDEsolver4.x_grid(), strike)
    v_vector5 = payoffs.put(PDEsolver5.x_grid(), strike)

    v_vector6 = payoffs.call(PDEsolver6.x_grid(), strike)
    v_vector7 = payoffs.call(PDEsolver7.x_grid(), strike)

plt.plot(PDEsolver1.x_grid(), v_vector1, 'k')

# Analytic result
if option == 'call':
    call = bs_call.Call(rate, vol, strike, expiry)
    plt.plot(PDEsolver1.x_grid(), call.price(PDEsolver1.x_grid(), 0), 'b')
elif option == 'put':
    put = bs_put.Put(rate, vol, strike, expiry)
    plt.plot(PDEsolver1.x_grid(), put.price(PDEsolver1.x_grid(), 0), 'b')

for t in range(t_steps):
    v_vector1 = PDEsolver1.propagation(dt, d_operator1, v_vector1)
    v_vector3 = PDEsolver3.propagation(dt, d_operator3, v_vector3)
    v_vector5 = PDEsolver5.propagation(dt, d_operator5, v_vector5)
    if t % ((t_steps - 1) // int(50 * t_max)) == 0:
        if exercise_type == 'American':
            if option == 'call':
                v_vector1 = np.maximum(v_vector1, PDEsolver1.x_grid() - strike)
                v_vector3 = np.maximum(v_vector3, PDEsolver3.x_grid() - strike)
                v_vector5 = np.maximum(v_vector5, PDEsolver5.x_grid() - strike)
            elif option == 'put':
                v_vector1 = np.maximum(v_vector1, strike - PDEsolver1.x_grid())
                v_vector3 = np.maximum(v_vector3, strike - PDEsolver3.x_grid())
                v_vector5 = np.maximum(v_vector5, strike - PDEsolver5.x_grid())

    v_vector6 = PDEsolver6.propagation(-dt, d_operator6, v_vector6)
    v_vector7 = PDEsolver7.propagation(-dt, d_operator7, v_vector7)

for t in range((t_steps - 1) // 2 + 1):
    v_vector2 = PDEsolver2.propagation(2 * dt, d_operator2, v_vector2)
    if t % (((t_steps - 1) // 2 + 1) // int(50 * t_max)) == 0:
        if exercise_type == 'American':
            if option == 'call':
                v_vector2 = np.maximum(v_vector2, PDEsolver2.x_grid() - strike)
            elif option == 'put':
                v_vector2 = np.maximum(v_vector2, strike - PDEsolver2.x_grid())
for t in range((t_steps - 1) // 4 + 1):
    v_vector4 = PDEsolver4.propagation(4 * dt, d_operator4, v_vector4)
    if t % (((t_steps - 1) // 4 + 1) // int(50 * t_max)) == 0:
        if exercise_type == 'American':
            if option == 'call':
                v_vector4 = np.maximum(v_vector4, PDEsolver4.x_grid() - strike)
            elif option == 'put':
                v_vector4 = np.maximum(v_vector4, strike - PDEsolver4.x_grid())
plt.plot(PDEsolver1.x_grid(), v_vector1, 'r')
plt.plot(PDEsolver2.x_grid(), v_vector2, 'b')
plt.plot(PDEsolver3.x_grid(), v_vector3, 'g')
plt.plot(PDEsolver4.x_grid(), v_vector4, '.b')
plt.plot(PDEsolver5.x_grid(), v_vector5, '.g')
plt.xlim((0, 120))
#plt.ylim((0, 8))
plt.show()

# Dupire result
plt.plot(PDEsolver6.x_grid(), v_vector6, 'r')
plt.plot(PDEsolver7.x_grid(), v_vector7, 'g')
call_prices1 = np.zeros(x_steps)
if option == 'call':
    call_dupire = bs_call.Call(rate, vol, strike, expiry)
else:
    call_dupire = bs_put.Put(rate, vol, strike, expiry)
for idx, k in enumerate(PDEsolver6.x_grid()):
    call_dupire.strike = k
    call_prices1[idx] = call_dupire.price(strike, 0)
plt.plot(PDEsolver6.x_grid()[::30], call_prices1[::30], '.b')
# plt.rcParams.update({'font.size': 45})
plt.tick_params(labelsize=16)
plt.xlim((20, 60))
plt.ylim((0, 25))
plt.xlabel('Strike', fontsize=16)
plt.ylabel('European call option price', fontsize=16)
# plt.savefig('Basic.png', bbox_inches="tight", dpi=200)
plt.show()

if option == 'put':
    s_points = [36, 38, 40, 42, 44]
    n_mc_paths = 1600000
    for x, v1, v2, exact in zip(PDEsolver1.x_grid(), v_vector1, v_vector2, put.price(PDEsolver1.x_grid(), 0)):
        for s in s_points:
            if math.fabs(x - s) < 1e-5:
                if option == 'call':
                    s_mc = call.path(s, expiry, n_mc_paths, antithetic=True)
                    s_mc = call.payoff(s_mc)
                elif option == 'put':
                    s_mc = put.path(s, expiry, n_mc_paths, antithetic=True)
                    s_mc = put.payoff(s_mc)
                s_mc *= math.exp(-rate * expiry)
                mean = sum(s_mc) / n_mc_paths
                n_half = n_mc_paths // 2
                std = math.sqrt(sum(((s_mc[:n_half] + s_mc[n_half:]) / 2 - mean) ** 2) / n_half)
                s_error = std / math.sqrt(n_half)
                print(x, v1, exact, math.fabs(v1 - exact) / v1, mean, std, s_error)
                dx = (x_max - x_min) / (x_steps - 1)
#               print(x, v1, exact, math.fabs(v1 - exact), math.fabs((v1 - v2) / 3 + (v1 - v_vector3[round(x / dx) // 2]) / 3))

    for x5, v5 in zip(PDEsolver5.x_grid(), v_vector5):
        dx = (x_max - x_min) / (x_steps - 1)
        v1 = v_vector1[round(x5 / dx)]
        v2 = v_vector2[round(x5 / dx)]
        v3 = v_vector3[round(x5 / dx) // 2]
        v4 = v_vector4[round(x5 / dx)]
        print(x5, (v2 - v4) / (v1 - v2), (v3 - v5) / (v1 - v3))

