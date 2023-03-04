import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from utils import misc


# Yield curve. Continuous compounded yield y(0,0,t).
time_grid = np.array([0.09, 0.26, 0.5, 1, 1.5, 2, 3, 4,
                      5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
yield_grid = np.array([-0.0034, 0.0005, 0.0061, 0.0135, 0.0179,
                       0.0202, 0.0224, 0.0237, 0.0246, 0.0252,
                       0.0256, 0.0261, 0.0265, 0.0270, 0.0277,
                       0.0281, 0.0267, 0.0249, 0.0233])
yield_curve = \
    misc.DiscreteFunc("yield", time_grid, yield_grid, interp_scheme="cubic")

# Discount curve.
disc_grid = np.exp(-yield_grid * time_grid)
# Time-grid extended to zero to avoid extrapolation error.
time_grid = np.append(0, time_grid)
disc_grid = np.append(1, disc_grid)
disc_curve = \
    misc.DiscreteFunc("discount", time_grid, disc_grid, interp_scheme="cubic")

# Speed of mean reversion strip.
time_grid = np.array([0, 10])
kappa_grid = 0.023 * np.array([1, 1])
kappa_strip = misc.DiscreteFunc("kappa", time_grid, kappa_grid)

# Volatility strip.
time_grid = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 20])
vol_grid = np.array([0.0165, 0.0143, 0.0140, 0.0132, 0.0128, 0.0103,
                     0.0067, 0.0096, 0.0087, 0.0091, 0.0098])
vol_strip = misc.DiscreteFunc("vol", time_grid, vol_grid)

# Extended yield curve.
time_grid = np.array(
    [0.025,
     0.282,
     0.531,
     0.780,
     1.029,
     1.279,
     1.530,
     1.780,
     2.026,
     2.278,
     2.533,
     2.776,
     3.025,
     3.277,
     3.532,
     3.776,
     4.025,
     4.277,
     4.537,
     4.775,
     5.024,
     5.276,
     5.533,
     5.782,
     6.032,
     6.281,
     6.533,
     6.782,
     7.028,
     7.277,
     7.532,
     7.776,
     8.025,
     8.277,
     8.531,
     8.775,
     9.024,
     9.276,
     9.531,
     9.777,
     10.026,
     10.278,
     10.535,
     10.776,
     11.026,
     11.283,
     11.532,
     11.781,
     12.030,
     12.280,
     12.531,
     12.778,
     13.027,
     13.276,
     13.531,
     13.777,
     14.026,
     14.278,
     14.533,
     14.776,
     15.026,
     15.278,
     15.538,
     15.776,
     16.025,
     16.277,
     16.534,
     16.775,
     17.024,
     17.282,
     17.531,
     17.783,
     18.029,
     18.278,
     18.530,
     18.777,
     19.026,
     19.278,
     19.529,
     19.776,
     20.025,
     20.277,
     20.529,
     20.775,
     21.024,
     21.276,
     21.528,
     21.777,
     22.026,
     22.284,
     22.533,
     22.782,
     23.031,
     23.280,
     23.530,
     23.779,
     24.028,
     24.277,
     24.529,
     24.775,
     25.024,
     25.276,
     25.528,
     25.777,
     26.027,
     26.278,
     26.530,
     26.777,
     27.026,
     27.278,
     27.535,
     27.776,
     28.025,
     28.283])
yield_grid = np.array([
    -0.00496,
    0.00092,
    0.00638,
    0.01056,
    0.01368,
    0.01610,
    0.01797,
    0.01931,
    0.02027,
    0.02103,
    0.02162,
    0.02207,
    0.02247,
    0.02282,
    0.02315,
    0.02344,
    0.02371,
    0.02397,
    0.02422,
    0.02442,
    0.02461,
    0.02478,
    0.02493,
    0.02506,
    0.02518,
    0.02530,
    0.02542,
    0.02554,
    0.02565,
    0.02577,
    0.02589,
    0.02600,
    0.02611,
    0.02622,
    0.02634,
    0.02644,
    0.02655,
    0.02666,
    0.02676,
    0.02686,
    0.02697,
    0.02707,
    0.02718,
    0.02729,
    0.02739,
    0.02749,
    0.02759,
    0.02767,
    0.02776,
    0.02783,
    0.02789,
    0.02795,
    0.02800,
    0.02804,
    0.02808,
    0.02810,
    0.02812,
    0.02812,
    0.02812,
    0.02812,
    0.02810,
    0.02807,
    0.02804,
    0.02800,
    0.02795,
    0.02790,
    0.02784,
    0.02778,
    0.02771,
    0.02764,
    0.02756,
    0.02748,
    0.02740,
    0.02732,
    0.02723,
    0.02715,
    0.02706,
    0.02697,
    0.02688,
    0.02679,
    0.02669,
    0.02660,
    0.02651,
    0.02642,
    0.02633,
    0.02624,
    0.02615,
    0.02606,
    0.02597,
    0.02587,
    0.02579,
    0.02570,
    0.02561,
    0.02552,
    0.02543,
    0.02534,
    0.02525,
    0.02517,
    0.02508,
    0.02499,
    0.02491,
    0.02482,
    0.02473,
    0.02465,
    0.02456,
    0.02448,
    0.02439,
    0.02431,
    0.02423,
    0.02415,
    0.02406,
    0.02399,
    0.02391,
    0.02383])
yield_curve_ext = \
    misc.DiscreteFunc("yield", time_grid, yield_grid, interp_scheme="cubic")

# Discount curve.
disc_grid = np.exp(-yield_grid * time_grid)
# Time-grid extended to zero to avoid extrapolation error.
time_grid_ext = np.append(0, time_grid)
disc_grid_ext = np.append(1, disc_grid)
disc_curve_ext = \
    misc.DiscreteFunc("discount", time_grid_ext, disc_grid_ext,
                      interp_scheme="cubic")

# Instantaneous forward rate curve. f(0,t).
forward_rate_grid = np.array([
    -0.00435,
    0.00731,
    0.01682,
    0.02171,
    0.02498,
    0.02702,
    0.02759,
    0.02739,
    0.02717,
    0.02697,
    0.02682,
    0.02679,
    0.02697,
    0.02724,
    0.02750,
    0.02775,
    0.02800,
    0.02819,
    0.02829,
    0.02827,
    0.02816,
    0.02803,
    0.02798,
    0.02802,
    0.02814,
    0.02831,
    0.02849,
    0.02869,
    0.02890,
    0.02912,
    0.02934,
    0.02954,
    0.02974,
    0.02993,
    0.03011,
    0.03025,
    0.03037,
    0.03051,
    0.03070,
    0.03093,
    0.03121,
    0.03147,
    0.03167,
    0.03180,
    0.03187,
    0.03187,
    0.03181,
    0.03168,
    0.03149,
    0.03125,
    0.03098,
    0.03069,
    0.03037,
    0.03002,
    0.02964,
    0.02924,
    0.02881,
    0.02835,
    0.02785,
    0.02735,
    0.02681,
    0.02626,
    0.02571,
    0.02523,
    0.02474,
    0.02426,
    0.02379,
    0.02337,
    0.02295,
    0.02254,
    0.02215,
    0.02179,
    0.02145,
    0.02112,
    0.02081,
    0.02053,
    0.02026,
    0.02001,
    0.01978,
    0.01958,
    0.01939,
    0.01921,
    0.01903,
    0.01886,
    0.01869,
    0.01852,
    0.01835,
    0.01818,
    0.01802,
    0.01786,
    0.01770,
    0.01754,
    0.01739,
    0.01724,
    0.01709,
    0.01694,
    0.01680,
    0.01666,
    0.01652,
    0.01638,
    0.01625,
    0.01612,
    0.01599,
    0.01587,
    0.01576,
    0.01566,
    0.01556,
    0.01547,
    0.01538,
    0.01530,
    0.01522,
    0.01516,
    0.01509,
    0.01504])
forward_rate_ext = \
    misc.DiscreteFunc("forward rate", time_grid, forward_rate_grid,
                      interp_scheme="cubic")


if __name__ == '__main__':

    # Plot yield curve.
    plt.plot(yield_curve.time_grid, 100 * yield_curve.values, "ob")
    plt.plot(yield_curve_ext.time_grid, 100 * yield_curve_ext.values, "-b")
    plt.xlabel("t [years]")
    plt.ylabel("y(0,0,t) [%]")
    plt.show()

    # Compare interpolation schemes for the yield curve...
    yield_interpolation = yield_curve.interpolation(yield_curve_ext.time_grid)
    yield_diff = yield_interpolation - yield_curve_ext.values
    plt.plot(yield_curve_ext.time_grid, 100 * yield_diff, "-b")
    plt.xlabel("t [years]")
    plt.ylabel("y_interpolated(0,0,t) - y(0,0,t) [%]")
    plt.show()

    # Compare curves
    time_grid_plot = 0.025 * np.arange(1201)

    yield_curve_plot = yield_curve.interpolation(time_grid_plot)

    yield_spline = \
        UnivariateSpline(yield_curve.time_grid, yield_curve.values, s=0)
    yield_spline = yield_spline.derivative()
    forward_rate_plot = \
        yield_curve_plot + time_grid_plot * yield_spline(time_grid_plot)

    ax1 = plt.subplot()
    plt.xlabel("t [years]")
    plt.xlim([0, 30])

    yield_curve_plot *= 100
    p1 = ax1.plot(time_grid_plot, yield_curve_plot,
                  "-b", label="y(0,0,t): Yield curve")

    forward_rate_ext_plot = forward_rate_ext.interpolation(time_grid_plot)
    forward_rate_ext_plot *= 100
#    p2 = ax1.plot(time_grid_plot, forward_rate_ext_plot, "-r", label="f(0,t)")

    forward_rate_plot *= 100
    p2 = ax1.plot(time_grid_plot, forward_rate_plot,
                  "-r", label="f(0,t): Instantaneous forward rate curve")
    plt.ylabel("y(0,0,t) and f(0,t) [%]")

    ax2 = ax1.twinx()
    disc_curve_plot = disc_curve.interpolation(time_grid_plot)
    p3 = ax2.plot(time_grid_plot, disc_curve_plot,
                  "-k", label="P(0,t): Discount curve")
    plt.ylabel("P(0,t)")
    plt.ylim([0, 1.05])

    plots = p1 + p2 + p3
    ax2.legend(plots, [legend.get_label() for legend in plots], loc=4)

    plt.show()
