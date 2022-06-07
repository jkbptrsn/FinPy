import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.vasicek.sde as sde
import models.vasicek.call as call
import models.vasicek.put as put
import models.vasicek.zcbond as zcbond
import utils.misc as misc


class SDE(unittest.TestCase):
    # TODO: test of Monte-Carlo error

    def test_getters_and_setters(self):
        """Test getters and setters of Vasicek SDE class."""
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        vasicek_sde = sde.SDE(kappa, mean_rate, vol)
        event_grid = np.arange(10)
        vasicek_sde.event_grid = event_grid
        self.assertEqual(vasicek_sde.kappa, kappa)
        self.assertEqual(vasicek_sde.mean_rate, mean_rate)
        self.assertEqual(vasicek_sde.vol, vol)
        self.assertTrue(np.array_equal(vasicek_sde.event_grid, event_grid))
        kappa += 1
        mean_rate += 1
        vol += 1
        event_grid += 1
        self.assertNotEqual(vasicek_sde.kappa, kappa)
        self.assertNotEqual(vasicek_sde.mean_rate, mean_rate)
        self.assertNotEqual(vasicek_sde.vol, vol)
        self.assertFalse(np.array_equal(vasicek_sde.event_grid, event_grid))
        vasicek_sde.kappa = kappa
        vasicek_sde.mean_rate = mean_rate
        vasicek_sde.vol = vol
        vasicek_sde.event_grid = event_grid
        self.assertEqual(vasicek_sde.kappa, kappa)
        self.assertEqual(vasicek_sde.mean_rate, mean_rate)
        self.assertEqual(vasicek_sde.vol, vol)
        self.assertTrue(np.array_equal(vasicek_sde.event_grid, event_grid))

    def test_zero_coupon_bond_pricing(self):
        """Monte-Carlo pricing of zero-coupon bond."""
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        spot = 0.02
        spot_vector = np.arange(-5, 6, 1) * spot
        t_max = 10
        vasicek_sde = sde.SDE(kappa, mean_rate, vol)
        event_grid = np.array([0, t_max])
        vasicek_sde.event_grid = event_grid
        vasicek_sde.initialization()
        np.random.seed(0)
        n_paths = 20000
        bond = zcbond.ZCBond(kappa, mean_rate, vol, t_max)
        for s in spot_vector:
            analytical = bond.price(s, 0)
            rates, discounts = vasicek_sde.paths(s, n_paths)
            numerical = np.sum(np.exp(discounts[-1, :])) / n_paths
            relative_diff = abs((numerical - analytical) / analytical)
            self.assertTrue(relative_diff < 0.01)

    def test_call_option_pricing(self):
        """Monte-Carlo pricing of European call option written on
        zero-coupon bond.
        """
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        spot = 0.02
        spot_vector = np.arange(-5, 6, 1) * spot
        t_max = 10
        strike = 0.7
        expiry = t_max / 2
        vasicek_sde = sde.SDE(kappa, mean_rate, vol)
        event_grid = np.array([0, expiry])
        vasicek_sde.event_grid = event_grid
        vasicek_sde.initialization()
        bond = zcbond.ZCBond(kappa, mean_rate, vol, t_max)
        call_option = call.Call(kappa, mean_rate, vol, strike, expiry, t_max)
        np.random.seed(0)
        n_paths = 20000
        for s in spot_vector:
            analytical = call_option.price(s, 0)
            rates, discounts = vasicek_sde.paths(s, n_paths)
            numerical = \
                np.maximum(bond.price(rates[-1, :], expiry) - strike, 0)
            numerical = np.sum(np.exp(discounts[-1, :]) * numerical) / n_paths
            relative_diff = abs((numerical - analytical) / analytical)
            self.assertTrue(relative_diff < 0.03)

    def test_put_option_pricing(self):
        """Monte-Carlo pricing of European put option written on
        zero-coupon bond.
        """
        kappa = 0.1
        mean_rate = 0.03
        vol = 0.05
        spot = 0.02
        spot_vector = np.arange(-5, 6, 1) * spot
        t_max = 10
        strike = 0.7
        expiry = t_max / 2
        vasicek_sde = sde.SDE(kappa, mean_rate, vol)
        event_grid = np.array([0, expiry])
        vasicek_sde.event_grid = event_grid
        vasicek_sde.initialization()
        bond = zcbond.ZCBond(kappa, mean_rate, vol, t_max)
        put_option = put.Put(kappa, mean_rate, vol, strike, expiry, t_max)
        np.random.seed(0)
        n_paths = 20000
        for s in spot_vector:
            analytical = put_option.price(s, 0)
            rates, discounts = vasicek_sde.paths(s, n_paths)
            numerical = \
                np.maximum(strike - bond.price(rates[-1, :], expiry), 0)
            numerical = np.sum(np.exp(discounts[-1, :]) * numerical) / n_paths
            relative_diff = abs((numerical - analytical) / analytical)
            self.assertTrue(relative_diff < 0.06)


if __name__ == '__main__':

    kappa_ = 0.1
    mean_rate_ = 0.03
    vol_ = 0.05
    spot_ = 0.02
    spot_vector_ = np.arange(-5, 6, 1) * spot_
    t_max_ = 10
    strike_ = 1.1
    expiry_ = t_max_ / 2
    # SDE object
    vasicek_sde = sde.SDE(kappa_, mean_rate_, vol_)
    # Zero-coupon bond
    bond = zcbond.ZCBond(kappa_, mean_rate_, vol_, t_max_)
    bond_price_n = spot_vector_ * 0
    bond_price_n_std = spot_vector_ * 0
    bond_price_a = spot_vector_ * 0
    bond_new = zcbond.ZCBond(kappa_, mean_rate_, vol_, t_max_)
    # Call option
    call_option = call.Call(kappa_, mean_rate_, vol_, strike_, expiry_, t_max_)
    call_price_n = spot_vector_ * 0
    call_price_n_std = spot_vector_ * 0
    call_price_a = spot_vector_ * 0
    # Put option
    put_option = put.Put(kappa_, mean_rate_, vol_, strike_, expiry_, t_max_)
    put_price_n = spot_vector_ * 0
    put_price_n_std = spot_vector_ * 0
    put_price_a = spot_vector_ * 0
    n_paths_ = 1000
    for idx, s in enumerate(spot_vector_):
        # Integration until t_max_
        vasicek_sde.event_grid = np.array([0, t_max_])
        vasicek_sde.initialization()
        rates, discounts = vasicek_sde.paths(s, n_paths_)
        # Price of bond with maturity = t_max_
        bond_price_a[idx] = bond.price(s, 0)
        bond_price_n[idx] = np.sum(np.exp(discounts[-1, :])) / n_paths_
        bond_price_n_std[idx] = \
            misc.monte_carlo_error(np.exp(discounts[-1, :]))
        # Integration until expiry_
        vasicek_sde.event_grid = np.array([0, expiry_])
        vasicek_sde.initialization()
        rates, discounts = vasicek_sde.paths(s, n_paths_)
        # Call option price
        call_price_a[idx] = call_option.price(s, 0)
        call_option_values = \
            np.maximum(bond_new.price(rates[-1, :], expiry_) - strike_, 0)
        call_price_n[idx] = \
            np.sum(np.exp(discounts[-1, :]) * call_option_values) / n_paths_
        call_price_n_std[idx] = misc.monte_carlo_error(np.exp(discounts[-1, :])
                                                       * call_option_values)
        # Put option price
        put_price_a[idx] = put_option.price(s, 0)
        put_option_values = \
            np.maximum(strike_ - bond_new.price(rates[-1, :], expiry_), 0)
        put_price_n[idx] = \
            np.sum(np.exp(discounts[-1, :]) * put_option_values) / n_paths_
        put_price_n_std[idx] = misc.monte_carlo_error(np.exp(discounts[-1, :])
                                                      * put_option_values)
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
