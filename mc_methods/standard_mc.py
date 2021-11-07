import math
import numpy as np
from typing import Tuple


class MonteCarlo:
    """Simple Monte-Carlo class for estimating option prices."""

    def __init__(self, n_paths, option):
        self._n_paths = n_paths
        self._option = option

    @property
    def n_paths(self):
        return self._n_paths

    @n_paths.setter
    def n_paths(self, val):
        self._n_paths = val

    @property
    def model_name(self):
        return self._option.model_name

    @property
    def option_type(self):
        return self._option.option_type

    def mc_price(self,
                 spot: (float, np.ndarray),
                 time: float,
                 antithetic: bool = False) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):
        """Monte-Carlo estimate of option price for each spot value."""
        time_to_maturity = self._option.expiry - time
        try:
            discount = math.exp(-self._option.rate * time_to_maturity)
        except AttributeError:
            discount = 1
        if antithetic:
            n_half = self.n_paths // 2
        if type(spot) is float:
            paths = self._option.path(spot, time_to_maturity,
                                      self.n_paths, antithetic=antithetic)
            payoff = discount * self._option.payoff(paths)
            mean = sum(payoff) / self.n_paths
            if antithetic:
                var = sum(((payoff[:n_half] + payoff[n_half:]) / 2
                           - mean) ** 2) / n_half
            else:
                var = sum((payoff - mean) ** 2) / self.n_paths
            return mean, math.sqrt(var)
        else:
            mean = np.ndarray(spot.shape[0])
            std = np.ndarray(spot.shape[0])
            for idx, s in enumerate(spot):
                paths = self._option.path(s, time_to_maturity,
                                          self.n_paths, antithetic=antithetic)
                payoff = discount * self._option.payoff(paths)
                mean[idx] = sum(payoff) / self.n_paths
                if antithetic:
                    var = sum(((payoff[:n_half] + payoff[n_half:]) / 2
                               - mean[idx]) ** 2) / n_half
                else:
                    var = sum((payoff - mean[idx]) ** 2) / self.n_paths
                std[idx] = math.sqrt(var)
            return mean, std
