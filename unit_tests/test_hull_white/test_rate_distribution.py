import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import zero_coupon_bond as zcbond
from models.hull_white import mc_andersen as mc_a
from models.hull_white import mc_pelsser as mc_p
from unit_tests.test_hull_white import input

plot_results = False
print_results = False


class JointDistributions(unittest.TestCase):
    """..."""

    def setUp(self) -> None:
        # Speed of mean reversion strip.
        self.kappa = input.kappa_strip
        # Volatility strip.
        self.vol = input.vol_strip
        # Discount curve.
        self.discount_curve = input.disc_curve
        # Number of MC paths.
        self.n_paths = 50000
        # Number of terms.
        self.n_terms = 21
        # Yearly frequency.
        self.frequency = 1
        # Event grid.
        self.event_grid = np.arange(self.n_terms) / self.frequency
        # SDE objects.
        self.mc_a = mc_a.SdeExactPiecewise(self.kappa,
                                           self.vol,
                                           self.discount_curve,
                                           self.event_grid)
        self.mc_p = mc_p.SdeExactPiecewise(self.kappa,
                                           self.vol,
                                           self.discount_curve,
                                           self.event_grid)
        # Zero-coupon bond objects.
        mat = self.n_terms - 1
        self.bond_a = zcbond.ZCBond(self.kappa,
                                    self.vol,
                                    self.discount_curve,
                                    mat,
                                    self.event_grid)
        self.bond_p = zcbond.ZCBondPelsser(self.kappa,
                                           self.vol,
                                           self.discount_curve,
                                           mat,
                                           self.event_grid)

    def test_compare(self):
        """..."""
        # MC paths using the Andersen transformation.
#        r_a, d_a = self.mc_a.paths(0, self.n_paths, seed=0)
#        r_a_adj = self.mc_a.rate_adjustment(r_a, self.bond_a.adjustment_rate)
#        d_a_adj = \
#            self.mc_a.discount_adjustment(d_a, self.bond_a.adjustment_discount)

        self.mc_a.paths(0, self.n_paths, seed=0)
        r_a_adj = self.mc_a.rate_adjustment(self.mc_a.rate_paths,
                                            self.bond_a.adjust_rate)
        d_a_adj = \
            self.mc_a.discount_adjustment(self.mc_a.discount_paths,
                                          self.bond_a.adjust_discount)

        # MC paths using the Pelsser transformation.
#        r_p, d_p = self.mc_p.paths(0, self.n_paths, seed=0)
        self.mc_p.paths(0, self.n_paths, seed=0)
        r_p = self.mc_p.rate_paths
        d_p = self.mc_p.discount_paths
        r_p_adj = self.mc_p.rate_adjustment(r_p, self.bond_p.adjust_rate)
        d_p_adj = \
            self.mc_p.discount_adjustment(d_p, self.bond_p.adjust_discount)

        if plot_results:
            bin_numbers = (50, 50)
            bin_range = [[-0.2, 0.2], [0, 3]]
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            fig.suptitle("Joint distributions")

            if False:
                r_a_plot = r_a_adj
                d_a_plot = d_a_adj
                r_p_plot = r_p_adj
                d_p_plot = d_p_adj
            else:
                r_a_plot = r_a
                d_a_plot = d_a
                r_p_plot = r_p
                d_p_plot = d_p

            for event_idx in range(1, self.n_terms):
                # Plot distribution based on Andersen transformation.
                hist_a, r_edge, d_edge = \
                    np.histogram2d(r_a_plot[event_idx, :],
                                   d_a_plot[event_idx, :],
                                   bins=bin_numbers,
                                   range=bin_range,
                                   density=True)
                r_middle = (r_edge[1:] + r_edge[:-1]) / 2
                d_middle = (d_edge[1:] + d_edge[:-1]) / 2
                r_mesh, d_mesh = np.meshgrid(r_middle, d_middle)
                first = ax[0].pcolormesh(r_mesh, d_mesh, hist_a.transpose())
                ax[0].set_xlabel("Short rate")
                ax[0].set_ylabel("Discount factor")
                ax[0].set_title("'Andersen'")
                cb1 = fig.colorbar(first, ax=ax[0])
                # Plot distribution based on Pelsser transformation.
                hist_p, _, _ = \
                    np.histogram2d(r_p_plot[event_idx, :],
                                   d_p_plot[event_idx, :],
                                   bins=bin_numbers,
                                   range=bin_range,
                                   density=True)
                second = ax[1].pcolormesh(r_mesh, d_mesh, hist_p.transpose())
                ax[1].set_xlabel("Short rate")
                ax[1].set_title("'Pelsser'")
                cb2 = fig.colorbar(second, ax=ax[1])
                # Plot difference.
                diff = hist_p.transpose() - hist_a.transpose()
                third = ax[2].pcolormesh(r_mesh, d_mesh, diff)
                ax[2].set_xlabel("Short rate")
                ax[2].set_title("'Pelsser' - 'Andersen'")
                cb3 = fig.colorbar(third, ax=ax[2])
                plt.pause(0.1)
                if event_idx != self.n_terms - 1:
                    cb1.remove()
                    cb2.remove()
                    cb3.remove()
            plt.show()

        # Compare paths.
        r_diff = np.abs(r_a_adj - r_p_adj)
        d_diff = np.abs(d_a_adj - d_p_adj)
        if print_results:
            print(np.max(r_diff), np.max(d_diff))
        self.assertTrue(np.max(r_diff) < 3.7e-16)
        self.assertTrue(np.max(d_diff) < 2.2e-14)
