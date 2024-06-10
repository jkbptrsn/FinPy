import typing

import numpy as np


def simple_forward_rate(
        bond_price_t2: typing.Union[float, np.ndarray],
        tau: float,
        bond_price_t1: typing.Union[float, np.ndarray] = 1.0) \
        -> typing.Union[float, np.ndarray]:
    """Calculate simple forward rate.

    The simple forward rate at time t in (t1, t2) is defined as:
        (1 + (t2 - t1) * forward_rate(t, t1, t2)) =
            bond_price_t1(t) / bond_price_t2(t).
    See Andersen & Piterbarg (2010), Section 4.1.

    Args:
        bond_price_t2: Price of zero-coupon bond with maturity t2.
        tau: Time interval between t1 and t2.
        bond_price_t1: Price of zero-coupon bond with maturity t1.
            Default is 1, in case that t = t1.

    Returns:
        Simple forward rate.
    """
    return (bond_price_t1 / bond_price_t2 - 1) / tau


def swap_schedule(
        fixing_start: int,
        fixing_end: int,
        fixing_frequency: int,
        events_per_fixing: int) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """Equidistant schedules for fixing, payment and event grids.

    Args:
        fixing_start: Year in which fixing starts.
        fixing_end: Year in which fixing ends.
        fixing_frequency: Yearly fixing frequency.
        events_per_fixing: Events per fixing period.

    Returns:
        Event grid, fixing and payment schedules.
    """
    # Number of events from time zero to fixing_end, both included.
    n_events = fixing_end * fixing_frequency * events_per_fixing + 1
    # Time step between two adjacent events.
    dt = fixing_end / (n_events - 1)
    # Equidistant event grid.
    event_grid = dt * np.arange(n_events)
    # Number of fixings from fixing_start to fixing_end.
    n_fixings = (fixing_end - fixing_start) * fixing_frequency
    # Fixing and payment schedules.
    fixing_schedule = np.zeros(n_fixings, dtype=int)
    payment_schedule = np.zeros(n_fixings, dtype=int)
    # Index of first fixing.
    start_idx = fixing_start * fixing_frequency * events_per_fixing
    for n in range(n_fixings):
        fixing_schedule[n] = start_idx + n * events_per_fixing
        payment_schedule[n] = fixing_schedule[n] + events_per_fixing
    return event_grid, fixing_schedule, payment_schedule
