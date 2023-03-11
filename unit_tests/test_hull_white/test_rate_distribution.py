import matplotlib.pyplot as plt
import numpy as np

from unit_tests.test_hull_white import input
from models.hull_white import call_option
from models.hull_white import caplet
from models.hull_white import put_option
from models.hull_white import sde
from models.hull_white import swap
from models.hull_white import swaption
from models.hull_white import zero_coupon_bond
from utils import misc
from utils import plots


if __name__ == '__main__':

    # Speed of mean reversion strip.
    kappa = input.kappa_strip
    # Volatility strip.
    vol = input.vol_strip
    # Discount curve.
    discount_curve = input.disc_curve

#    event_grid_plot = 0.1 * np.arange(0, 251, 250)
    event_grid_plot = 0.1 * np.arange(251)
    n_paths = 20000  # 200000
    hw = sde.SDE(kappa, vol, event_grid_plot, int_step_size=1 / 52)
    hw_p = sde.SDEPelsser(kappa, vol, event_grid_plot, int_step_size=1 / 52)

    rate, discount = hw.paths(0, n_paths, seed=0)
    rate_p, discount_p = hw_p.paths(0, n_paths, seed=0)

    _, mean = hw.calc_rate_mean_custom(0, 1)
    variance = hw.calc_rate_variance_custom(0, 1)
#    plots.plot_rate_distribution(1, rate, mean, np.sqrt(variance))
    _, mean_p = hw_p.calc_rate_mean_custom(0, 1)
    mean_p *= 0
    variance_p = hw_p.calc_rate_variance_custom(0, 1)
#    plots.plot_rate_distribution(1, rate_p, mean_p, np.sqrt(variance_p))

    for event_idx in range(1, 251, 10):
        _, mean = hw.calc_rate_mean_custom(0, event_idx)
        variance = hw.calc_rate_variance_custom(0, event_idx)
#        plots.plot_rate_distribution(event_idx, rate, mean, np.sqrt(variance))
        _, mean_p = hw_p.calc_rate_mean_custom(0, event_idx)
        mean_p *= 0
        variance_p = hw_p.calc_rate_variance_custom(0, event_idx)
#        plots.plot_rate_distribution(event_idx, rate_p, mean_p, np.sqrt(variance_p))

        plots.plot_rate_discount_distribution(event_idx, rate, discount)

        plt.cla()
