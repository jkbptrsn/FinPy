import numpy as np


def binary_asset_call(spot: (float, np.ndarray),
                      strike: float) -> (float, np.ndarray):
    """Long asset-or-nothing call option position."""
    return (spot > strike) * spot


def binary_asset_put(spot: (float, np.ndarray),
                     strike: float) -> (float, np.ndarray):
    """Long asset-or-nothing put option position."""
    return (spot < strike) * spot


def binary_cash_call(spot: (float, np.ndarray),
                     strike: float) -> (float, np.ndarray):
    """Long cash-or-nothing call option position."""
    return (spot > strike) * 1


def binary_cash_put(spot: (float, np.ndarray),
                    strike: float) -> (float, np.ndarray):
    """Long cash-or-nothing put option position."""
    return (spot < strike) * 1


def call(spot: (float, np.ndarray),
         strike: float) -> (float, np.ndarray):
    """Long call option position."""
    return np.maximum(spot - strike, 0)


def put(spot: (float, np.ndarray),
        strike: float) -> (float, np.ndarray):
    """Long put option position."""
    return np.maximum(strike - spot, 0)


def zero_coupon_bond(spot: (float, np.ndarray)) -> (float, np.ndarray):
    """Long zero coupon bond position."""

    # TODO: Use 100 instead

    return 0 * spot + 1
