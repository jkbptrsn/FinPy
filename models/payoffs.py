import numpy as np


def call(spot, strike):
    """
    Call option

    Parameters
    ----------
    spot : float / numpy.ndarray
    strike : float

    Returns
    -------
    float / numpy.ndarray
    """
    return np.maximum(spot - strike, 0)


def put(spot, strike):
    """
    Put option

    Parameters
    ----------
    spot : float / numpy.ndarray
    strike : float

    Returns
    -------
    float / numpy.ndarray
    """
    return np.maximum(strike - spot, 0)


def binary_cash_call(spot, strike):
    """
    Cash-or-nothing call option: Pays out one unit of cash if the spot
    is above the strike at expiry

    Parameters
    ----------
    spot : float / numpy.ndarray
    strike : float

    Returns
    -------
    float / numpy.ndarray
    """
    return (spot - strike > 0) * 1


def binary_asset_call(spot, strike):
    """
    Asset-or-nothing call option: Pays out one unit of the asset if the
    spot is above the strike at expiry

    Parameters
    ----------
    spot : float / numpy.ndarray
    strike : float

    Returns
    -------
    float / numpy.ndarray
    """
    return (spot - strike > 0) * spot


def binary_cash_put(spot, strike):
    """
    Cash-or-nothing put option: Pays out one unit of cash if the spot is
    below the strike at expiry

    Parameters
    ----------
    spot : float / numpy.ndarray
    strike : float

    Returns
    -------
    float / numpy.ndarray
    """
    return (spot - strike < 0) * 1


def binary_asset_put(spot, strike):
    """
    Asset-or-nothing put option: Pays out one unit of the asset if the
    spot is below the strike at expiry

    Parameters
    ----------
    spot : float / numpy.ndarray
    strike : float

    Returns
    -------
    float / numpy.ndarray
    """
    return (spot - strike < 0) * spot
