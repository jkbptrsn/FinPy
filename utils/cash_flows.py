import math

import numpy as np


def cash_flow(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        principal: float = 1,
        _type: str = "annuity",
        n_io_terms: int = 0) -> np.ndarray:
    """Cash flow on payment grid.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        principal: Loan principal. Default is 1.
        _type: Type of cash flow.
            - 'annuity'
            - 'deferred'
            - 'standing'
            - 'serial'
            Default is 'annuity'.
        n_io_terms: Number of 'interest only' terms for deferred
            annuities. Default is 0.

    Returns:
        Cash flow.
    """
    cf = cash_flow_split(
        coupon, frequency, payment_grid, principal, _type, n_io_terms)
    return cf.sum(axis=0)


def cash_flow_split(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        principal: float = 1,
        _type: str = "annuity",
        n_io_terms: int = 0) -> np.ndarray:
    """Cash flows for both redemption and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        principal: Loan principal. Default is 1.
        _type: Type of cash flow.
            - 'annuity'
            - 'deferred'
            - 'standing'
            - 'serial'
            Default is 'annuity'.
        n_io_terms: Number of 'interest only' terms for deferred
            annuities. Default is 0.

    Returns:
        Cash flows.
    """
    if _type == "annuity":
        return annuity(coupon, frequency, payment_grid, principal)
    elif _type == "deferred":
        return deferred_annuity(
            coupon, frequency, payment_grid, principal, n_io_terms)
    elif _type == "standing":
        return standing_loan(coupon, frequency, payment_grid, principal)
    elif _type == "serial":
        return serial_loan(coupon, frequency, payment_grid, principal)
    else:
        raise ValueError(f"Unknown cash flow type: {_type}")


###############################################################################


def cash_flow_issuance(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        n_issuance_terms: int,
        principal: float = 1,
        _type: str = "annuity",
        n_io_terms: int = 0) -> np.ndarray:
    """Cash flow on payment grid.

        Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        n_issuance_terms: Number of terms in issuance period.
        principal: Loan principal. Default is 1.
        _type: Type of cash flow.
            - 'annuity'
            - 'deferred'
            - 'standing'
            - 'serial'
            Default is 'annuity'.
        n_io_terms: Number of 'interest only' terms for deferred
            annuities. Default is 0.

    Returns:
        Cash flows.
    """
    cf = cash_flow_split_issuance(
        coupon, frequency, payment_grid, n_issuance_terms, principal, _type,
        n_io_terms)
    return cf.sum(axis=0)


def cash_flow_split_issuance(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        n_issuance_terms: int,
        principal: float = 1,
        _type: str = "annuity",
        n_io_terms: int = 0) -> np.ndarray:
    """Cash flows for both redemption and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        n_issuance_terms: Number of terms in issuance period.
        principal: Loan principal. Default is 1.
        _type: Type of cash flow.
            - 'annuity'
            - 'deferred'
            - 'standing'
            - 'serial'
            Default is 'annuity'.
        n_io_terms: Number of 'interest only' terms for deferred
            annuities. Default is 0.

    Returns:
        Cash flows.
    """
    # Number of payment terms per issuance.
    n_payments = payment_grid.size - n_issuance_terms
    cf = np.zeros((2, payment_grid.size))
    for idx in range(n_issuance_terms):
        # For issuance at idx, last payment at idx + n_payments.
        cf[:, idx:(n_payments + 1) + idx] += cash_flow_split(
            coupon, frequency, payment_grid[idx:(n_payments + 1) + idx],
            principal, _type, n_io_terms)
    # Normalization of redemption payments.
    cf[0] *= principal / cf[0].sum()
    # Determine interest payments.
    cf[1] = coupon * np.flip(np.cumsum(np.flip(cf[0]))) / frequency
    return cf


def set_payment_grid(
        t_initial: float,
        t_final: float,
        frequency: int) -> np.ndarray:
    """Set up grid with payment events.

    Args:
        t_initial: Initial time of first payment period.
        t_final: Final time of last payment period.
        frequency: Yearly payment frequency.

    Returns:
        Payment grid.
    """
    # Year-fraction between payment events.
    dt = 1 / frequency
    # Number of payment events.
    n_payments = (t_final - t_initial) / dt
    if 1 - n_payments < 1.0e-12:
        if abs(n_payments - round(n_payments)) < 1.0e-12:
            n_payments = round(n_payments)
        else:
            n_payments = math.floor(n_payments)
    else:
        raise ValueError("Mismatch between total payment period and "
                         "payment frequency.")
    return dt * np.arange(1, n_payments + 1) + t_initial


def set_payment_grid_issuance(
        t_initial: float,
        t_final: float,
        frequency: int,
        n_issuance_terms: int) -> np.ndarray:
    """Set up grid with payment events, including issuance period.

    Args:
        t_initial: Initial time of first payment period.
        t_final: Final time of last payment period.
        frequency: Yearly payment frequency.
        n_issuance_terms: Number of terms in issuance period.

    Returns:
        Payment grid.
    """
    payment_grid = set_payment_grid(t_initial, t_final, frequency)
    # Year-fraction between payment events.
    dt = 1 / frequency
    issuance_period = dt * np.arange(-(n_issuance_terms - 1), 1) + t_initial
    return np.append(issuance_period, payment_grid)


def set_deadline_grid(
        payment_grid: np.ndarray,
        deadline_step: float = 1 / 6,
        move_origin: bool = True) -> np.ndarray:
    """Set up grid with deadline events.

    Args:
        payment_grid: Grid of payment events.
        deadline_step: Time step between deadline event and
            corresponding payment event. Default is 1 / 6.
        move_origin: Move origin of deadline grid, if first deadline is
            passed. Default is True.

    Returns:
        Deadline grid.
    """
    deadline_grid = payment_grid - deadline_step
    if move_origin:
        if deadline_grid[0] < 0:
            deadline_grid[0] = 0
        if np.min(deadline_grid) < 0:
            raise ValueError("Deadline grid is ill defined.")
    return deadline_grid


def set_event_grid(
        payment_grid: np.ndarray,
        time_step: float = 0.01,
        *,
        deadline_grid: np.ndarray = None) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """Set up event grid, and payment and deadline schedules.

    Args:
        payment_grid: Grid of payment events.
        time_step: Time step between propagation events.
            Default is 0.01.
        deadline_grid: Grid of deadline event.s

    Returns:
        Event grid, payment schedule, deadline schedule.
    """
    event_grid = np.zeros(1)
    payment_schedule = np.zeros(payment_grid.size, dtype=int)
    if deadline_grid is None:
        deadline_grid = set_deadline_grid(payment_grid)
    deadline_schedule = np.zeros(deadline_grid.size, dtype=int)
    for idx, (deadline_time, payment_time) in \
            enumerate(zip(deadline_grid, payment_grid)):
        # Time of previous event.
        if idx == 0:
            time_previous_event = 0
        else:
            time_previous_event = payment_grid[idx - 1]
        # Number of propagation steps from previous event to current
        # deadline event.
        delta_time = deadline_time - time_previous_event
        n_steps = math.floor(delta_time / time_step)
        event_grid_tmp = event_grid[-1] + time_step * np.arange(1, n_steps + 1)
        event_grid = np.append(event_grid, event_grid_tmp)
        if delta_time - n_steps * time_step > 1.0e-12:
            dt = delta_time - n_steps * time_step
            event_grid = np.append(event_grid, event_grid[-1] + dt)
            n_steps += 1
        # Update schedules.
        deadline_schedule[idx:] += n_steps
        payment_schedule[idx:] += n_steps
        # Update time of previous event.
        time_previous_event = deadline_time
        # Number of propagation steps from previous event to current
        # payment event.
        delta_time = payment_time - time_previous_event
        n_steps = math.floor(delta_time / time_step)
        event_grid_tmp = event_grid[-1] + time_step * np.arange(1, n_steps + 1)
        event_grid = np.append(event_grid, event_grid_tmp)
        if delta_time - n_steps * time_step > 1.0e-12:
            dt = delta_time - n_steps * time_step
            event_grid = np.append(event_grid, event_grid[-1] + dt)
            n_steps += 1
        # Update schedules.
        deadline_schedule[idx + 1:] += n_steps
        payment_schedule[idx:] += n_steps
    return event_grid, payment_schedule, deadline_schedule


def annuity_factor(
        n_terms: int,
        coupon_term: float) -> float:
    """Calculate annuity factor ("alfahage").

    Hint: sum_{t = 1}^{n} 1 / (1 + r)^t = (1 - (1 + r)^{-n}) / r,
        using the sum formula
        sum_{i = 0}^{n - 1} x^i = (1 - x^n) / (1 - x).

    Args:
        n_terms: Number of terms, assumed equidistantly positioned.
        coupon_term: Coupon rate per term.

    Returns:
        Annuity factor.
    """
    return (1 - (1 + coupon_term) ** (-n_terms)) / coupon_term


def annuity_yield(
        n_terms: int,
        coupon_term: float,
        principal: float = 1) -> float:
    """Constant yield of annuity.

    Args:
        n_terms: Number of terms, assumed equidistantly positioned.
        coupon_term: Coupon rate per term.
        principal: Loan principal. Default is 1.

    Returns:
        Annuity yield.
    """
    return principal / annuity_factor(n_terms, coupon_term)


def annuity_interest(
        coupon_term: float,
        principal: float) -> float:
    """Interest payment for current loan principal.

    Args:
        coupon_term: Coupon rate per term.
        principal: Current loan principal.

    Returns:
        Interest payment.
    """
    return coupon_term * principal


def annuity_redemption(
        coupon_term: float,
        principal: float,
        _yield: float) -> float:
    """Redemption payment for current loan principal.

    Args:
        coupon_term: Coupon rate per term.
        principal: Current loan principal.
        _yield: Annuity yield.

    Returns:
        Redemption payment.
    """
    return _yield - annuity_interest(coupon_term, principal)


def annuity(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        principal: float = 1) -> np.ndarray:
    """Cash flow of annuity.

    Cash flow is split in redemption and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = payment_grid.size
    cf = np.zeros((2, n_payments))
    # Coupon rate per term.
    coupon_term = coupon / frequency
    # Annuity yield.
    _yield = annuity_yield(n_payments, coupon_term, principal)
    # Remaining principal before payment.
    remaining_principal = principal
    for idx in range(n_payments):
        # Redemption payment.
        cf[0, idx] = annuity_redemption(
            coupon_term, remaining_principal, _yield)
        # Interest payment.
        cf[1, idx] = annuity_interest(coupon_term, remaining_principal)
        # Subtract redemption payment from principal.
        remaining_principal -= cf[0, idx]
    return cf


def deferred_annuity(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        principal: float = 1,
        n_io_terms: int = 0) -> np.ndarray:
    """Cash flow of deferred annuity.

    Cash flow is split in redemption and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        principal: Loan principal.
        n_io_terms: Number of 'interest only' terms. Default is 0.

    Returns:
        Cash flow.
    """
    # Number of interest payments.
    n_payments = payment_grid.size
    cf = np.zeros((2, n_payments))
    # Coupon rate per term.
    coupon_term = coupon / frequency
    # Number of redemption payments.
    n_redemption = n_payments - n_io_terms
    # Annuity yield.
    _yield = annuity_yield(n_redemption, coupon_term, principal)
    # Interest payments in initial 'interest only' period.
    cf[1, :n_io_terms] = coupon_term * principal
    # Remaining principal before payment.
    remaining_principal = principal
    for idx in range(n_io_terms, n_payments):
        # Redemption payment.
        cf[0, idx] = annuity_redemption(
            coupon_term, remaining_principal, _yield)
        # Interest payment.
        cf[1, idx] = annuity_interest(coupon_term, remaining_principal)
        # Subtract redemption payment from principal.
        remaining_principal -= cf[0, idx]
    return cf


###############################################################################


def standing_loan(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        principal: float = 1) -> np.ndarray:
    """Cash flow of standing loan with fixed rate.

    Cash flow is split in redemption and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = payment_grid.size
    cf = np.zeros((2, n_payments))
    # Constant interest payment per term.
    cf[1, :] = coupon * principal / frequency
    # Principal redeemed at maturity.
    cf[0, -1] = principal
    return cf


def serial_loan(
        coupon: float,
        frequency: int,
        payment_grid: np.ndarray,
        principal: float = 1) -> np.ndarray:
    """Cash flow of serial loan with fixed rate.

    Cash flow is split in redemption and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment events.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = payment_grid.size
    cf = np.zeros((2, n_payments))
    # Constant redemption payment per term.
    cf[0, :] = principal / n_payments
    # Coupon rate per term.
    coupon_term = coupon / frequency
    # Interest payment at each term.
    cf[1] = coupon_term * np.flip(np.cumsum(np.flip(cf[0])))
    return cf


###############################################################################


def print_cash_flow(cf: np.ndarray) -> None:
    """Print cash flow to screen.

    Args:
        cf: Cash flow.
    """
    if len(cf.shape) == 1:
        for idx, total in enumerate(cf):
            print(f"Term {idx + 1:4}:  "
                  f"Total ={total:10.5f}")
        print(f"Summation:  "
              f"Total ={cf.sum():10.5f}")
    else:
        for idx, (redemption, interest) in enumerate(zip(cf[0, :], cf[1, :])):
            print(f"Term {idx + 1:4}:  "
                  f"Redemption ={redemption:10.5f},  "
                  f"Interest ={interest:10.5f},  "
                  f"Total ={redemption + interest:10.5f}")
        print(f"Summation:  "
              f"Redemption ={cf[0, :].sum():10.5f},  "
              f"Interest ={cf[1, :].sum():10.5f},  "
              f"Total ={cf.sum():10.5f}")
