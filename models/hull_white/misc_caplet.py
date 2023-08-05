import math
import typing

import numpy as np
from scipy.stats import norm

from utils.global_types import Instrument


def caplet_price(spot: typing.Union[float, np.ndarray],
                 strike_rate: float,
                 tenor: float,
                 event_idx: int,
                 fixing_idx: int,
                 payment_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 option_type: Instrument = Instrument.CAPLET) \
        -> typing.Union[float, np.ndarray]:
    """Calculate caplet/floorlet price.

    Price of caplet or floorlet written on simple forward rate.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike_rate: Caplet or floorlet rate.
        tenor: Time between fixing and payment events.
        event_idx: Index on event grid.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        option_type: Caplet or floorlet. Default is caplet.

    Returns:
        Caplet/floorlet price.
    """
    if option_type == Instrument.CAPLET:
        omega = -1
    elif option_type == Instrument.FLOORLET:
        omega = 1
    else:
        raise ValueError(f"Unknown instrument type: {option_type}")
    # P(t,t_fixing).
    zcbond.mat_idx = fixing_idx
    price1 = zcbond.price(spot, event_idx)
    # P(t,t_payment).
    zcbond.mat_idx = payment_idx
    price2 = zcbond.price(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike_rate, tenor, v)
    factor = (1 + strike_rate * tenor)
    return omega * (factor * price2 * norm.cdf(omega * d_plus)
                    - price1 * norm.cdf(omega * d_minus))


def caplet_delta(spot: typing.Union[float, np.ndarray],
                 strike_rate: float,
                 tenor: float,
                 event_idx: int,
                 fixing_idx: int,
                 payment_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 option_type: Instrument = Instrument.CAPLET) \
        -> typing.Union[float, np.ndarray]:
    """Calculate caplet/floorlet delta.

    Delta of caplet or floorlet written on simple forward rate.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike_rate: Caplet or floorlet rate.
        tenor: Time between fixing and payment events.
        event_idx: Index on event grid.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        option_type: Caplet or floorlet. Default is caplet.

    Returns:
        Caplet/floorlet delta.
    """
    if option_type == Instrument.CAPLET:
        omega = -1
    elif option_type == Instrument.FLOORLET:
        omega = 1
    else:
        raise ValueError(f"Unknown instrument type: {option_type}")
    # P(t,t_fixing).
    zcbond.mat_idx = fixing_idx
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    # P(t,t_payment).
    zcbond.mat_idx = payment_idx
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike_rate, tenor, v)
    factor = (1 + strike_rate * tenor)
    # 1st order spatial derivative of d-function at time t.
    d_delta = dd_dr(price1, delta1, price2, delta2, v)
    first_term = factor * delta2 * norm.cdf(omega * d_plus) \
        - delta1 * norm.cdf(omega * d_minus)
    second_term = factor * price2 * norm.pdf(omega * d_plus) \
        - price1 * norm.pdf(omega * d_minus)
    second_term *= d_delta
    return omega * first_term + omega ** 2 * second_term


def caplet_gamma(spot: typing.Union[float, np.ndarray],
                 strike_rate: float,
                 tenor: float,
                 event_idx: int,
                 fixing_idx: int,
                 payment_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 option_type: Instrument = Instrument.CAPLET) \
        -> typing.Union[float, np.ndarray]:
    """Calculate caplet/floorlet gamma.

    Gamma of caplet or floorlet written on simple forward rate.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike_rate: Caplet or floorlet rate.
        tenor: Time between fixing and payment events.
        event_idx: Index on event grid.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        option_type: Caplet or floorlet. Default is caplet.

    Returns:
        Caplet/floorlet gamma.
    """
    if option_type == Instrument.CAPLET:
        omega = -1
    elif option_type == Instrument.FLOORLET:
        omega = 1
    else:
        raise ValueError(f"Unknown instrument type: {option_type}")
    # P(t,t_fixing).
    zcbond.mat_idx = fixing_idx
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    gamma1 = zcbond.gamma(spot, event_idx)
    # P(t,t_payment).
    zcbond.mat_idx = payment_idx
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)
    gamma2 = zcbond.gamma(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike_rate, tenor, v)
    factor = (1 + strike_rate * tenor)
    # 1st order spatial derivative of d-function at time t.
    d_delta = dd_dr(price1, delta1, price2, delta2, v)
    # 2nd order spatial derivative of d-function at time t.
    d_gamma = d2d_dr2(price1, delta1, gamma1, price2, delta2, gamma2, v)
    first_term = \
        factor * gamma2 * norm.cdf(omega * d_plus) \
        + factor * delta2 * norm.pdf(omega * d_plus) * omega * d_delta \
        - gamma1 * norm.cdf(omega * d_minus) \
        - delta1 * norm.pdf(omega * d_minus) * omega * d_delta
    second_term = \
        factor * delta2 * norm.pdf(omega * d_plus) * d_delta \
        + factor * price2 * norm.pdf(omega * d_plus) \
        * (d_gamma - d_plus * d_delta ** 2) \
        - delta1 * norm.pdf(omega * d_minus) * d_delta \
        - price1 * norm.pdf(omega * d_minus) \
        * (d_gamma - d_minus * d_delta ** 2)
    return omega * first_term + omega ** 2 * second_term


def caplet_theta(spot: typing.Union[float, np.ndarray],
                 strike_rate: float,
                 tenor: float,
                 event_idx: int,
                 fixing_idx: int,
                 payment_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 dv_dt_eg: np.ndarray,
                 option_type: Instrument = Instrument.CAPLET) \
        -> typing.Union[float, np.ndarray]:
    """Calculate caplet/floorlet theta.

    Theta of caplet or floorlet written on simple forward rate.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike_rate: Caplet or floorlet rate.
        tenor: Time between fixing and payment events.
        event_idx: Index on event grid.
        fixing_idx: Fixing index on event grid.
        payment_idx: Payment index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        dv_dt_eg: dv_dt-function on event grid.
        option_type: Caplet or floorlet. Default is caplet.

    Returns:
        Caplet/floorlet theta.
    """
    if option_type == Instrument.CAPLET:
        omega = -1
    elif option_type == Instrument.FLOORLET:
        omega = 1
    else:
        raise ValueError(f"Unknown instrument type: {option_type}")
    # P(t,t_fixing).
    zcbond.mat_idx = fixing_idx
    price1 = zcbond.price(spot, event_idx)
    theta1 = zcbond.theta(spot, event_idx)
    # P(t,t_payment).
    zcbond.mat_idx = payment_idx
    price2 = zcbond.price(spot, event_idx)
    theta2 = zcbond.theta(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # dv_dt-function at time t.
    dv_dt = dv_dt_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike_rate, tenor, v)
    factor = (1 + strike_rate * tenor)
    # 1st order time derivative of d-functions at time t.
    d_theta = dd_dt(price1, theta1, price2, theta2,
                    strike_rate, tenor, v, dv_dt)
    first_term = factor * theta2 * norm.cdf(omega * d_plus) \
        - theta1 * norm.cdf(omega * d_minus)
    second_term = factor * price2 * norm.pdf(omega * d_plus) * d_theta[0] \
        - price1 * norm.pdf(omega * d_minus) * d_theta[1]
    return omega * first_term + omega ** 2 * second_term


def d_function(price1: typing.Union[float, np.ndarray],
               price2: typing.Union[float, np.ndarray],
               strike_rate: float,
               tenor: float,
               v: float) -> tuple:
    """Calculate d-functions.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        strike_rate: Caplet or floorlet rate.
        tenor: Time between fixing and payment events.
        v: Value of v-function at time t.

    Returns:
        d-functions.
    """
    d = np.log((1 + strike_rate * tenor) * price2 / price1)
    d_plus = (d + v / 2) / math.sqrt(v)
    d_minus = (d - v / 2) / math.sqrt(v)
    return d_plus, d_minus


def dd_dr(price1: typing.Union[float, np.ndarray],
          delta1: typing.Union[float, np.ndarray],
          price2: typing.Union[float, np.ndarray],
          delta2: typing.Union[float, np.ndarray],
          v: float) -> typing.Union[float, np.ndarray]:
    """Calculate 1st order spatial derivative of d-functions.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        delta1: Delta of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        delta2: Delta of zero-coupon bond price with maturity T*.
        v: Value of v-function at time t.

    Returns:
        1st order spatial derivative of d-functions.
    """
    return (delta2 / price2 - delta1 / price1) / math.sqrt(v)


def d2d_dr2(price1: typing.Union[float, np.ndarray],
            delta1: typing.Union[float, np.ndarray],
            gamma1: typing.Union[float, np.ndarray],
            price2: typing.Union[float, np.ndarray],
            delta2: typing.Union[float, np.ndarray],
            gamma2: typing.Union[float, np.ndarray],
            v: float) -> typing.Union[float, np.ndarray]:
    """Calculate 2nd order spatial derivative of d-functions.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        delta1: Delta of zero-coupon bond price with maturity T.
        gamma1: Gamma of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        delta2: Delta of zero-coupon bond price with maturity T*.
        gamma2: Gamma of zero-coupon bond price with maturity T*.
        v: Value of v-function at time t.

    Returns:
        2nd order spatial derivative of d-functions.
    """
    return (gamma2 / price2 - delta2 ** 2 / price2 ** 2
            - gamma1 / price1 + delta1 ** 2 / price1 ** 2) / math.sqrt(v)


def dd_dt(price1: typing.Union[float, np.ndarray],
          theta1: typing.Union[float, np.ndarray],
          price2: typing.Union[float, np.ndarray],
          theta2: typing.Union[float, np.ndarray],
          strike_rate: float,
          tenor: float,
          v: float,
          dv_dt: float) -> tuple:
    """Calculate 1st order time derivative of d-functions.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        theta1: Theta of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        theta2: Theta of zero-coupon bond price with maturity T*.
        strike_rate: Caplet or floorlet rate.
        tenor: Time between fixing and payment events.
        v: Value of v-function at time t.
        dv_dt: Value of dv_dt-function at time t.

    Returns:
        1st order time derivative of d-functions.
    """
    d_plus, d_minus = d_function(price1, price2, strike_rate, tenor, v)
    factor = dv_dt / (2 * v)
    term = theta2 / price2 - theta1 / price1
    dd_plus_dt = -factor * d_plus + (term + dv_dt / 2) / math.sqrt(v)
    dd_minus_dt = -factor * d_minus + (term - dv_dt / 2) / math.sqrt(v)
    return dd_plus_dt, dd_minus_dt
