import matplotlib.pyplot as plt
import numpy as np

from models.vasicek import sde
from models.vasicek import european_option
from models.vasicek import zero_coupon_bond
from utils import misc


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
    monte_carlo = sde.SdeExact(kappa_, mean_rate_, vol_, event_grid_)
    # Zero-coupon bond
    bond = zero_coupon_bond.ZCBond(kappa_, mean_rate_, vol_, maturity_idx_,
                                   event_grid_)
    bond_price_a = spot_vector_ * 0
    bond_price_n = spot_vector_ * 0
    bond_price_n_error = spot_vector_ * 0
    # Call option
    call = european_option.EuropeanOption(kappa_, mean_rate_, vol_, strike_,
                                          expiry_idx_, maturity_idx_,
                                          event_grid_, "Call")
    call_price_a = spot_vector_ * 0
    call_price_n = spot_vector_ * 0
    call_price_n_error = spot_vector_ * 0
    # Put option
    put = european_option.EuropeanOption(kappa_, mean_rate_, vol_, strike_,
                                         expiry_idx_, maturity_idx_,
                                         event_grid_, "Put")
    put_price_a = spot_vector_ * 0
    put_price_n = spot_vector_ * 0
    put_price_n_error = spot_vector_ * 0
    # Initialize random number generator
    rng_ = np.random.default_rng(0)
    # Number of paths for each Monte-Carlo estimate
    n_paths_ = 1000
    for idx, s in enumerate(spot_vector_):
        # Price of bond with maturity = maturity_
#        _, discounts = monte_carlo.paths(s, n_paths_, rng=rng_)
        monte_carlo.paths(s, n_paths_, rng=rng_)
        discounts = monte_carlo.discount_paths

        bond_price_a[idx] = bond.price(s, 0)
        bond_price_n[idx] = discounts[maturity_idx_, :].mean()
        bond_price_n_error[idx] = \
            misc.monte_carlo_error(discounts[maturity_idx_, :])
        # Call option price with expiry = expiry_
#        rates, discounts = monte_carlo.paths(s, n_paths_, rng=rng_)
        monte_carlo.paths(s, n_paths_, rng=rng_)
        rates, discounts = monte_carlo.rate_paths, monte_carlo.discount_paths

        call_price_a[idx] = call.price(s, 0)
        call_option_values = \
            np.maximum(bond.price(rates[expiry_idx_, :], expiry_idx_)
                       - strike_, 0)
        call_option_values *= discounts[expiry_idx_, :]
        call_price_n[idx] = call_option_values.mean()
        call_price_n_error[idx] = misc.monte_carlo_error(call_option_values)
        # Put option price with expiry = expiry_
#        rates, discounts = monte_carlo.paths(s, n_paths_, rng=rng_)
        monte_carlo.paths(s, n_paths_, rng=rng_)
        rates, discounts = monte_carlo.rate_paths, monte_carlo.discount_paths

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
