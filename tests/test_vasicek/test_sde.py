import matplotlib.pyplot as plt
import numpy as np
import unittest

from models.vasicek import sde
from models.vasicek import call_option
from models.vasicek import put_option
from models.vasicek import zero_coupon_bond
from utils import misc


class SDE(unittest.TestCase):

    def test_zero_coupon_bond_pricing(self):
        """Monte-Carlo estimate of zero-coupon bond price.

        Comparison of Monte-Carlo estimate and analytical formula.
        """
        # Model parameters
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        # Spor rate
        spot = 0.02
        spot_vector = np.arange(-4, 5, 2) * spot
        maturity = 10
        event_grid = np.array([0, maturity])
        maturity_idx = 1
        # SDE object
        monte_carlo = sde.SDE(kappa, mean_rate, vol, event_grid)
        # Zero-coupon bond
        bond = zero_coupon_bond.ZCBond(kappa, mean_rate, vol, event_grid,
                                       maturity_idx)
        # Initialize random number generator
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate
        n_paths = 5000
        # Number of repetitions of Monte-Carlo simulation
        n_rep = 100
        for s in spot_vector:
            # Analytical result
            price_a = bond.price(s, 0)
            # Numerical result; no variance reduction
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = monte_carlo.paths(s, n_paths, rng=rng)
                price_n = discounts[maturity_idx, :].mean()
                error[rep] += abs((price_n - price_a) / price_a)
            # print(s, price_a, error.mean(), error.std())
            self.assertTrue(error.mean() < 0.009 and error.std() < 0.007)
            # Numerical result; Antithetic sampling
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = \
                    monte_carlo.paths(s, n_paths, rng=rng, antithetic=True)
                price_n = discounts[maturity_idx, :].mean()
                error[rep] += abs((price_n - price_a) / price_a)
            # print(s, price_a, error.mean(), error.std())
            self.assertTrue(error.mean() < 0.006 and error.std() < 0.004)

    def test_call_option_pricing(self):
        """Monte-Carlo estimate of European call option price.

        Comparison of Monte-Carlo estimate and analytical formula.
        """
        # Model parameters
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        # Spor rate
        spot = 0.02
        spot_vector = np.arange(-4, 5, 2) * spot
        expiry = 5
        maturity = 10
        event_grid = np.array([0, expiry, maturity])
        expiry_idx = 1
        maturity_idx = 2
        strike = 1.1
        # SDE object
        monte_carlo = sde.SDE(kappa, mean_rate, vol, event_grid)
        # Zero-coupon bond
        bond = zero_coupon_bond.ZCBond(kappa, mean_rate, vol, event_grid, maturity_idx)
        # European call option written on zero-coupon bond
        call = call_option.Call(kappa, mean_rate, vol, event_grid, strike,
                                expiry_idx, maturity_idx)
        # Initialize random number generator
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate
        n_paths = 5000
        # Number of repetitions of Monte-Carlo simulation
        n_rep = 100
        for s in spot_vector:
            # Analytical result
            price_a = call.price(s, 0)
            # Numerical result; no variance reduction
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = monte_carlo.paths(s, n_paths, rng=rng)
                price_n = \
                    np.maximum(bond.price(rates[expiry_idx, :], expiry_idx)
                               - strike, 0)
                price_n *= discounts[expiry_idx, :]
                price_n = price_n.mean()
                error[rep] = abs((price_n - price_a) / price_a)
            # print(s, price_a, error.mean(), error.std())
            self.assertTrue(error.mean() < 0.04 and error.std() < 0.031)
            # Numerical result; Antithetic sampling
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = \
                    monte_carlo.paths(s, n_paths, rng=rng, antithetic=True)
                price_n = \
                    np.maximum(bond.price(rates[expiry_idx, :], expiry_idx)
                               - strike, 0)
                price_n *= discounts[expiry_idx, :]
                price_n = price_n.mean()
                error[rep] = abs((price_n - price_a) / price_a)
            # print(s, price_a, error.mean(), error.std())
            self.assertTrue(error.mean() < 0.043 and error.std() < 0.032)

    def test_put_option_pricing(self):
        """Monte-Carlo estimate of European put option price.

        Comparison of Monte-Carlo estimate and analytical formula.
        """
        # Model parameters
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        # Spor rate
        spot = 0.02
        spot_vector = np.arange(-4, 5, 2) * spot
        expiry = 5
        maturity = 10
        event_grid = np.array([0, expiry, maturity])
        expiry_idx = 1
        maturity_idx = 2
        strike = 1.1
        # SDE object
        monte_carlo = sde.SDE(kappa, mean_rate, vol, event_grid)
        # Zero-coupon bond
        bond = zero_coupon_bond.ZCBond(kappa, mean_rate, vol, event_grid,
                                       maturity_idx)
        # European put option written on zero-coupon bond
        put = put_option.Put(kappa, mean_rate, vol, event_grid, strike,
                             expiry_idx, maturity_idx)
        # Initialize random number generator
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate
        n_paths = 5000
        # Number of repetitions of Monte-Carlo simulation
        n_rep = 100
        for s in spot_vector:
            # Analytical result
            price_a = put.price(s, 0)
            # Numerical result; no variance reduction
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = monte_carlo.paths(s, n_paths, rng=rng)
                price_n = np.maximum(strike - bond.price(rates[expiry_idx, :],
                                                         expiry_idx), 0)
                price_n *= discounts[expiry_idx, :]
                price_n = price_n.mean()
                error[rep] = abs((price_n - price_a) / price_a)
            # print(s, price_a, error.mean(), error.std())
            self.assertTrue(error.mean() < 0.016 and error.std() < 0.013)
            # Numerical result; Antithetic sampling
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = \
                    monte_carlo.paths(s, n_paths, rng=rng, antithetic=True)
                price_n = np.maximum(strike - bond.price(rates[expiry_idx, :],
                                                         expiry_idx), 0)
                price_n *= discounts[expiry_idx, :]
                price_n = price_n.mean()
                error[rep] = abs((price_n - price_a) / price_a)
            # print(s, price_a, error.mean(), error.std())
            self.assertTrue(error.mean() < 0.012 and error.std() < 0.01)


if __name__ == '__main__':

    # Model parameters
    kappa_ = 0.1
    mean_rate_ = 0.03
    vol_ = 0.05
    # Spor rate
    spot_ = 0.02
    spot_vector_ = np.arange(-5, 6, 1) * spot_
    expiry_ = 5
    maturity_ = 10
    event_grid_ = np.array([0, expiry_, maturity_])
    expiry_idx_ = 1
    maturity_idx_ = 2
    strike_ = 1.1
    # SDE object
    monte_carlo = sde.SDE(kappa_, mean_rate_, vol_, event_grid_)
    # Zero-coupon bond
    bond = zero_coupon_bond.ZCBond(kappa_, mean_rate_, vol_, event_grid_,
                                   maturity_idx_)
    bond_price_a = spot_vector_ * 0
    bond_price_n = spot_vector_ * 0
    bond_price_n_error = spot_vector_ * 0
    # Call option
    call = call_option.Call(kappa_, mean_rate_, vol_, event_grid_, strike_,
                            expiry_idx_, maturity_idx_)
    call_price_a = spot_vector_ * 0
    call_price_n = spot_vector_ * 0
    call_price_n_error = spot_vector_ * 0
    # Put option
    put = put_option.Put(kappa_, mean_rate_, vol_, event_grid_, strike_,
                         expiry_idx_, maturity_idx_)
    put_price_a = spot_vector_ * 0
    put_price_n = spot_vector_ * 0
    put_price_n_error = spot_vector_ * 0
    # Initialize random number generator
    rng_ = np.random.default_rng(0)
    # Number of paths for each Monte-Carlo estimate
    n_paths_ = 1000
    for idx, s in enumerate(spot_vector_):
        # Price of bond with maturity = maturity_
        _, discounts = monte_carlo.paths(s, n_paths_, rng=rng_)
        bond_price_a[idx] = bond.price(s, 0)
        bond_price_n[idx] = discounts[maturity_idx_, :].mean()
        bond_price_n_error[idx] = \
            misc.monte_carlo_error(discounts[maturity_idx_, :])
        # Call option price with expiry = expiry_
        rates, discounts = monte_carlo.paths(s, n_paths_, rng=rng_)
        call_price_a[idx] = call.price(s, 0)
        call_option_values = \
            np.maximum(bond.price(rates[expiry_idx_, :], expiry_idx_)
                       - strike_, 0)
        call_option_values *= discounts[expiry_idx_, :]
        call_price_n[idx] = call_option_values.mean()
        call_price_n_error[idx] = misc.monte_carlo_error(call_option_values)
        # Put option price with expiry = expiry_
        rates, discounts = monte_carlo.paths(s, n_paths_, rng=rng_)
        put_price_a[idx] = put.price(s, 0)
        put_option_values = \
            np.maximum(strike_
                       - bond.price(rates[expiry_idx_, :], expiry_idx_), 0)
        put_option_values *= discounts[expiry_idx_, :]
        put_price_n[idx] = put_option_values.mean()
        put_price_n_error[idx] = misc.monte_carlo_error(put_option_values)
    # Plot error bars corresponding to 95%-confidence intervals
    bond_price_n_error *= 1.96
    call_price_n_error *= 1.96
    put_price_n_error *= 1.96
    plt.plot(spot_vector_, bond_price_a, "-b", label="Zero-coupon bond")
    plt.errorbar(spot_vector_, bond_price_n, bond_price_n_error,
                 linestyle="none", marker="o", color="b", capsize=5)
    plt.plot(spot_vector_, call_price_a, "-r", label="Call option")
    plt.errorbar(spot_vector_, call_price_n, call_price_n_error,
                 linestyle="none", marker="o", color="r", capsize=5)
    plt.plot(spot_vector_, put_price_a, "-g", label="Put option")
    plt.errorbar(spot_vector_, put_price_n, put_price_n_error,
                 linestyle="none", marker="o", color="g", capsize=5)
    plt.title(f"95% confidence intervals ({n_paths_} samples)")
    plt.xlabel("Spot rate")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
