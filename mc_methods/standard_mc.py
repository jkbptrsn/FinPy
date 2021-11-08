import math
import numpy as np
from typing import Tuple


class MonteCarlo:
    """Monte-Carlo class for calculating option prices and greeks."""

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

    def price(self,
              spot: (float, np.ndarray),
              time: float,
              antithetic: bool = False) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):
        """Calculation of option price for each spot value. Returns
        Monte-Carlo mean and standard deviation."""
        time_to_maturity = self._option.expiry - time
        try:
            discount = math.exp(-self._option.rate * time_to_maturity)
        except AttributeError:
            discount = 1
        if type(spot) is float:
            paths = self._option.path(spot, time_to_maturity,
                                      self.n_paths, antithetic)
            payoff = discount * self._option.payoff(paths)
            mean = sum(payoff) / self.n_paths
            if antithetic:
                n_half = self.n_paths // 2
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
                                          self.n_paths, antithetic)
                payoff = discount * self._option.payoff(paths)
                mean[idx] = sum(payoff) / self.n_paths
                if antithetic:
                    n_half = self.n_paths // 2
                    var = sum(((payoff[:n_half] + payoff[n_half:]) / 2
                               - mean[idx]) ** 2) / n_half
                else:
                    var = sum((payoff - mean[idx]) ** 2) / self.n_paths
                std[idx] = math.sqrt(var)
            return mean, std

    def greek(self,
              spot: (float, np.ndarray),
              time: float,
              greek: str = 'delta',
              method: str = 'path-wise',
              antithetic: bool = False) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):
        """Calculation of greek for each spot value. Returns Monte-Carlo
        mean and standard deviation."""
        time_to_maturity = self._option.expiry - time
        try:
            discount = math.exp(-self._option.rate * time_to_maturity)
        except AttributeError:
            discount = 1
        if type(spot) is float:
            if method == 'path-wise':
                paths, factors = \
                    self._option.path_wise(spot, time_to_maturity,
                                           self.n_paths, greek, antithetic)
                derivative = \
                    discount * self._option.payoff_dds(paths) * factors
                mean = sum(derivative) / self.n_paths
                if antithetic:
                    n_half = self.n_paths // 2
                    var = sum(((derivative[:n_half] + derivative[n_half:]) / 2
                               - mean) ** 2) / n_half
                else:
                    var = sum((derivative - mean) ** 2) / self.n_paths
                return mean, math.sqrt(var)
        else:
            mean = np.ndarray(spot.shape[0])
            std = np.ndarray(spot.shape[0])
            if method == 'path-wise':
                for idx, s in enumerate(spot):
                    paths, factors = \
                        self._option.path_wise(s, time_to_maturity,
                                               self.n_paths, greek, antithetic)
                    derivative \
                        = discount * self._option.payoff_dds(paths) * factors
                    mean[idx] = sum(derivative) / self.n_paths
                    if antithetic:
                        n_half = self.n_paths // 2
                        var = sum(((derivative[:n_half]
                                    + derivative[n_half:]) / 2
                                   - mean[idx]) ** 2) / n_half
                    else:
                        var = sum((derivative - mean[idx]) ** 2) / self.n_paths
                    std[idx] = math.sqrt(var)
                return mean, std
