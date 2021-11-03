import numpy as np


def call(spot: (float, np.ndarray), strike: float) -> (float, np.ndarray):
    """Payoff of long call option position."""
    return np.maximum(spot - strike, 0)


def put(spot: (float, np.ndarray), strike: float) -> (float, np.ndarray):
    """Payoff of long put option position."""
    return np.maximum(strike - spot, 0)


def binary_cash_call(spot: (float, np.ndarray), strike: float) \
        -> (float, np.ndarray):
    """Payoff of long cash-or-nothing call option position."""
    return (spot - strike > 0) * 1


def binary_asset_call(spot: (float, np.ndarray), strike: float) \
        -> (float, np.ndarray):
    """Payoff of long asset-or-nothing call option position."""
    return (spot - strike > 0) * spot


def binary_cash_put(spot: (float, np.ndarray), strike: float) \
        -> (float, np.ndarray):
    """Payoff of long cash-or-nothing put option position."""
    return (strike - spot > 0) * 1


def binary_asset_put(spot: (float, np.ndarray), strike: float) \
        -> (float, np.ndarray):
    """Payoff of long asset-or-nothing put option position."""
    return (strike - spot > 0) * spot
