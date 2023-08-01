import numpy as np


def swap_schedule(fixing_start: int,
                  fixing_end: int,
                  fixing_frequency: int,
                  events_per_fixing: int) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """...

    Args:
        fixing_start: Year in which fixing starts.
        fixing_end: Year in which fixing ends.
        fixing_frequency: Yearly fixing frequency.
        events_per_fixing: Events per fixing period.

    Returns:
        ...
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
