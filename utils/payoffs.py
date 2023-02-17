import numpy as np
from typing import Union


def binary_asset_call(spot: Union[float, np.ndarray],
                      strike: float) -> Union[float, np.ndarray]:
    """Long asset-or-nothing call option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot > strike) * spot


def binary_asset_put(spot: Union[float, np.ndarray],
                     strike: float) -> Union[float, np.ndarray]:
    """Long asset-or-nothing put option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot < strike) * spot


def binary_cash_call(spot: Union[float, np.ndarray],
                     strike: float) -> Union[float, np.ndarray]:
    """Long cash-or-nothing call option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot > strike) * 1


def binary_cash_put(spot: Union[float, np.ndarray],
                    strike: float) -> Union[float, np.ndarray]:
    """Long cash-or-nothing put option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot < strike) * 1


def call(spot: Union[float, np.ndarray],
         strike: float) -> Union[float, np.ndarray]:
    """Long call option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return np.maximum(spot - strike, 0)


def put(spot: Union[float, np.ndarray],
        strike: float) -> Union[float, np.ndarray]:
    """Long put option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return np.maximum(strike - spot, 0)


def zero_coupon_bond(spot: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """Long zero-coupon bond position.

    Args:
        spot: Current rate.

    Returns:
        Payoff.
    """
    return 1 + 0 * spot
