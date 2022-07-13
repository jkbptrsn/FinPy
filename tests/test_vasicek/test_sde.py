import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.vasicek.sde as sde
import models.vasicek.call as call
import models.vasicek.put as put
import models.vasicek.zero_coupon_bond as zcbond
import utils.misc as misc


class SDE(unittest.TestCase):
    # TODO: test of Monte-Carlo error

    def test_zero_coupon_bond_pricing(self):
        """Compare Monte-Carlo pricing of zero-coupon bond with
        analytical result.
        """
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        spot = 0.02
        spot_vector = np.arange(-5, 6, 1) * spot
        maturity = 10
        event_grid = np.array([0, maturity])
        maturity_idx = 1
        vasicek_sde = sde.SDE(kappa, mean_rate, vol, event_grid)
        vasicek_sde.initialization()
        np.random.seed(0)
        n_paths = 30000
        bond = zcbond.ZCBond(kappa, mean_rate, vol, event_grid, maturity_idx)
        for s in spot_vector:
            # Analytical result
            price_a = bond.price(s, 0)
            # Numerical result; no variance reduction
            rates, discounts = vasicek_sde.paths(s, n_paths)
            price_n = np.sum(discounts[maturity_idx, :]) / n_paths
            diff = abs((price_n - price_a) / price_a)
            self.assertTrue(diff < 0.01)
            # Numerical result; Antithetic sampling
            rates, discounts = vasicek_sde.paths(s, n_paths, antithetic=True)
            price_n = np.sum(discounts[maturity_idx, :]) / n_paths
            diff = abs((price_n - price_a) / price_a)
            self.assertTrue(diff < 0.005)

    def test_call_option_pricing(self):
        """Compare Monte-Carlo pricing of European call option written
        on zero-coupon bond with analytical result.
        """
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        spot = 0.02
        spot_vector = np.arange(-5, 6, 1) * spot
        maturity = 10
        expiry = maturity / 2
        strike = 0.7
        event_grid = np.array([0, expiry, maturity])
        expiry_idx = 1
        maturity_idx = 2
        vasicek_sde = sde.SDE(kappa, mean_rate, vol, event_grid)
        vasicek_sde.initialization()
        bond = zcbond.ZCBond(kappa, mean_rate, vol, event_grid, maturity_idx)
        call_option = call.Call(kappa, mean_rate, vol, event_grid,
                                strike, expiry_idx, maturity_idx)
        np.random.seed(0)
        n_paths = 30000
        for s in spot_vector:
            # Analytical result
            price_a = call_option.price(s, 0)
            # Numerical result; no variance reduction
            rates, discounts = vasicek_sde.paths(s, n_paths)
            price_n = np.maximum(bond.price(rates[expiry_idx, :], expiry)
                                 - strike, 0)
            price_n = np.sum(discounts[expiry_idx, :] * price_n) / n_paths
            diff = abs((price_n - price_a) / price_a)
            self.assertTrue(diff < 0.02)
            # Numerical result; Antithetic sampling
            rates, discounts = vasicek_sde.paths(s, n_paths, antithetic=True)
            price_n = np.maximum(bond.price(rates[expiry_idx, :], expiry)
                                 - strike, 0)
            price_n = np.sum(discounts[expiry_idx, :] * price_n) / n_paths
            diff = abs((price_n - price_a) / price_a)
            self.assertTrue(diff < 0.01)

    def test_put_option_pricing(self):
        """Compare Monte-Carlo pricing of European put option written on
        zero-coupon bond with analytical result.
        """
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        spot = 0.02
        spot_vector = np.arange(-5, 6, 1) * spot
        maturity = 10
        expiry = maturity / 2
        strike = 0.7
        event_grid = np.array([0, expiry, maturity])
        expiry_idx = 1
        maturity_idx = 2
        vasicek_sde = sde.SDE(kappa, mean_rate, vol, event_grid)
        vasicek_sde.initialization()
        bond = zcbond.ZCBond(kappa, mean_rate, vol, event_grid, maturity_idx)
        put_option = put.Put(kappa, mean_rate, vol, event_grid,
                             strike, expiry_idx, maturity_idx)
        np.random.seed(0)
        n_paths = 30000
        for s in spot_vector:
            # Analytical result
            price_a = put_option.price(s, 0)
            # Numerical result; no variance reduction
            rates, discounts = vasicek_sde.paths(s, n_paths)
            price_n = \
                np.maximum(strike
                           - bond.price(rates[expiry_idx, :], expiry), 0)
            price_n = np.sum(discounts[expiry_idx, :] * price_n) / n_paths
            diff = abs((price_n - price_a) / price_a)
            self.assertTrue(diff < 0.04)
            # Numerical result; Antithetic sampling
            rates, discounts = vasicek_sde.paths(s, n_paths, antithetic=True)
            price_n = \
                np.maximum(strike
                           - bond.price(rates[expiry_idx, :], expiry), 0)
            price_n = np.sum(discounts[expiry_idx, :] * price_n) / n_paths
            diff = abs((price_n - price_a) / price_a)
            self.assertTrue(diff < 0.02)


if __name__ == '__main__':

    kappa_ = 0.1
    mean_rate_ = 0.03
    vol_ = 0.05
    spot_ = 0.02
    spot_vector_ = np.arange(-5, 6, 1) * spot_
    maturity_ = 10
    expiry_ = maturity_ / 2
    maturity_idx_ = 2
    expiry_idx_ = 1
    strike_ = 1.1
    # Event grid
    event_grid = np.array([0, expiry_, maturity_])
    # SDE object
    vasicek_sde = sde.SDE(kappa_, mean_rate_, vol_, event_grid)
    # Zero-coupon bond
    bond = zcbond.ZCBond(kappa_, mean_rate_, vol_, event_grid, maturity_idx_)
    bond_price_n = spot_vector_ * 0
    bond_price_n_std = spot_vector_ * 0
    bond_price_a = spot_vector_ * 0
    bond_new = \
        zcbond.ZCBond(kappa_, mean_rate_, vol_, event_grid, maturity_idx_)
    # Call option
    call_option = call.Call(kappa_, mean_rate_, vol_, event_grid,
                            strike_, expiry_idx_, maturity_idx_)
    call_price_n = spot_vector_ * 0
    call_price_n_std = spot_vector_ * 0
    call_price_a = spot_vector_ * 0
    # Put option
    put_option = put.Put(kappa_, mean_rate_, vol_, event_grid,
                         strike_, expiry_idx_, maturity_idx_)
    put_price_n = spot_vector_ * 0
    put_price_n_std = spot_vector_ * 0
    put_price_a = spot_vector_ * 0
    np.random.seed(0)
    n_paths_ = 100
    for idx, s in enumerate(spot_vector_):
        # Integration until maturity_
        vasicek_sde.event_grid = np.array([0, maturity_])
        vasicek_sde.initialization()
        rates, discounts = vasicek_sde.paths(s, n_paths_, antithetic=True)
        # Price of bond with maturity = maturity_
        bond_price_a[idx] = bond.price(s, 0)
        bond_price_n[idx] = np.sum(discounts[-1, :]) / n_paths_
        bond_price_n_std[idx] = misc.monte_carlo_error(discounts[-1, :])
        # Integration until expiry_
        vasicek_sde.event_grid = np.array([0, expiry_])
        vasicek_sde.initialization()
        rates, discounts = vasicek_sde.paths(s, n_paths_, antithetic=True)
        # Call option price
        call_price_a[idx] = call_option.price(s, 0)
        call_option_values = \
            np.maximum(bond_new.price(rates[-1, :], expiry_) - strike_, 0)
        call_price_n[idx] = \
            np.sum(discounts[-1, :] * call_option_values) / n_paths_
        call_price_n_std[idx] = \
            misc.monte_carlo_error(discounts[-1, :] * call_option_values)
        # Put option price
        put_price_a[idx] = put_option.price(s, 0)
        put_option_values = \
            np.maximum(strike_ - bond_new.price(rates[-1, :], expiry_), 0)
        put_price_n[idx] = \
            np.sum(discounts[-1, :] * put_option_values) / n_paths_
        put_price_n_std[idx] = \
            misc.monte_carlo_error(discounts[-1, :] * put_option_values)
    plt.plot(spot_vector_, bond_price_a, '-b', label="Zero coupon bond")
    plt.errorbar(spot_vector_, bond_price_n, np.transpose(bond_price_n_std),
                 linestyle="none", marker="o", color="b", capsize=5)
    plt.plot(spot_vector_, call_price_a, '-r', label="Call option")
    plt.errorbar(spot_vector_, call_price_n, np.transpose(call_price_n_std),
                 linestyle="none", marker="o", color="r", capsize=5)
    plt.plot(spot_vector_, put_price_a, '-g', label="Put option")
    plt.errorbar(spot_vector_, put_price_n, np.transpose(put_price_n_std),
                 linestyle="none", marker="o", color="g", capsize=5)
    plt.xlabel("Spot rate")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    unittest.main()
