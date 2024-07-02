import numpy as np
import typing


def binary_asset_call(
        spot: typing.Union[float, np.ndarray],
        strike: float) -> typing.Union[float, np.ndarray]:
    """Long asset-or-nothing call option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot > strike) * spot


def binary_asset_put(
        spot: typing.Union[float, np.ndarray],
        strike: float) -> typing.Union[float, np.ndarray]:
    """Long asset-or-nothing put option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot < strike) * spot


def binary_cash_call(
        spot: typing.Union[float, np.ndarray],
        strike: float) -> typing.Union[float, np.ndarray]:
    """Long cash-or-nothing call option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot > strike) * 1


def binary_cash_put(
        spot: typing.Union[float, np.ndarray],
        strike: float) -> typing.Union[float, np.ndarray]:
    """Long cash-or-nothing put option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return (spot < strike) * 1


def call(spot: typing.Union[float, np.ndarray],
         strike: float) -> typing.Union[float, np.ndarray]:
    """Long call option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return np.maximum(spot - strike, 0)


def put(spot: typing.Union[float, np.ndarray],
        strike: float) -> typing.Union[float, np.ndarray]:
    """Long put option position.

    Args:
        spot: Current price of underlying asset.
        strike: Strike price of underlying asset.

    Returns:
        Payoff.
    """
    return np.maximum(strike - spot, 0)


def zero_coupon_bond(spot: typing.Union[float, np.ndarray]) \
        -> typing.Union[float, np.ndarray]:
    """Long zero-coupon bond position.

    Args:
        spot: Current short rate.

    Returns:
        Payoff.
    """
    return 1 + 0 * spot
