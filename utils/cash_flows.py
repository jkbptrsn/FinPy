import math

import numpy as np


def cash_flow(coupon: float,
              frequency: int,
              cf_grid: np.ndarray,
              principal: float = 1,
              _type: str = "annuity",
              io: int = 0) -> np.ndarray:
    """Cash flow.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        cf_grid: Grid of cash flow events.
        principal: Loan principal.
        _type: Type of cash flow. 'annuity', 'deferred', 'standing' or
            'serial'. Default is 'annuity'.
        io: Number of 'interest only' terms for deferred annuities.
            Default is 0.

    Returns:
        Cash flow.
    """
    cf = cash_flow_split(coupon, frequency, cf_grid, principal, _type, io)
    return cf.sum(axis=0)


def cash_flow_split(coupon: float,
                    frequency: int,
                    cf_grid: np.ndarray,
                    principal: float = 1,
                    _type: str = "annuity",
                    io: int = 0) -> np.ndarray:
    """Cash flows for both installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        cf_grid: Grid of cash flow events.
        principal: Loan principal.
        _type: Type of cash flow. 'annuity', 'deferred', 'standing' or
            'serial'. Default is 'annuity'.
        io: Number of 'interest only' terms for deferred annuities.
            Default is 0.

    Returns:
        Cash flows.
    """
    if _type == "annuity":
        return annuity(coupon, frequency, cf_grid, principal)
    elif _type == "deferred":
        return deferred_annuity(coupon, frequency, cf_grid, principal, io)
    elif _type == "standing":
        return standing_loan(coupon, frequency, cf_grid, principal)
    elif _type == "serial":
        return serial_loan(coupon, frequency, cf_grid, principal)
    else:
        raise ValueError(f"Cash flow type is unknown: {_type}")


def set_cash_flow_grid(t_initial: float,
                       t_final: float,
                       frequency: int) -> np.ndarray:
    """Set up grid with 'cash flow' events.

    Args:
        t_initial: Initial time of first payment period.
        t_final: Final time of last payment period.
        frequency: Yearly payment frequency.

    Returns:
        'Cash flow' grid.
    """
    # Year-fraction between payment dates.
    dt = 1 / frequency
    # Number of payment dates.
    n_payments = (t_final - t_initial) / dt
    if 1 - n_payments < 1.0e-12:
        if abs(n_payments - round(n_payments)) < 1.0e-12:
            n_payments = round(n_payments)
        else:
            n_payments = math.floor(n_payments)
    else:
        raise ValueError("Mismatch in total payment period and "
                         "payment frequency.")
    return dt * np.arange(1, n_payments + 1) + t_initial


def set_event_grid(cf_grid: np.ndarray,
                   time_step: float = 0.01) -> (np.ndarray, np.ndarray):
    """Set up event grid, and cash flow schedule.

    Args:
        cf_grid: Grid of cash flow events.
        time_step: Time step between events. Default is 0.01.

    Returns:
        Event grid, cash flow schedule.
    """
    event_grid = np.zeros(1)
    cash_flow_schedule = np.zeros(cf_grid.size, dtype=int)
    cf_schedule_append = np.append(0, cf_grid)
    for idx, cf_time_step in enumerate(np.diff(cf_schedule_append)):
        n_steps = math.floor(cf_time_step / time_step)
        event_grid_tmp = event_grid[-1] + time_step * np.arange(1, n_steps + 1)
        event_grid = np.append(event_grid, event_grid_tmp)
        if cf_time_step - n_steps * time_step > 1.0e-12:
            dt = cf_time_step - n_steps * time_step
            event_grid = np.append(event_grid, event_grid[-1] + dt)
            cash_flow_schedule[idx] = n_steps + 1
        else:
            cash_flow_schedule[idx] = n_steps
    return event_grid, np.cumsum(cash_flow_schedule)


def annuity_factor(n_terms: int,
                   coupon: float) -> float:
    """Calculate annuity factor ("alfahage").

    Hint: \sum_{t = 1}^{n} 1 / (1 + r)^t = (1 - (1 + r)^{-n}) / r,
        using the sum formula
            \sum_{i = 0}^{n - 1} x^i = (1 - x^n) / (1 - x).

    Args:
        n_terms: Number of terms, assumed equidistantly positioned.
        coupon: Coupon rate per term.

    Returns:
        Annuity factor.
    """
    return (1 - (1 + coupon) ** (-n_terms)) / coupon


def annuity_yield(n_terms: int,
                  coupon: float,
                  principal: float = 1) -> float:
    """Constant yield of annuity.

    Args:
        n_terms: Number of terms, assumed equidistantly positioned.
        coupon: Coupon rate per term.
        principal: Loan principal. Default is 1.

    Returns:
        Annuity yield.
    """
    return principal / annuity_factor(n_terms, coupon)


def annuity_interest(coupon_term: float,
                     principal: float) -> float:
    """Interest payment for current loan principal.

    Args:
        coupon_term: Coupon rate per term.
        principal: Current loan principal.

    Returns:
        Interest payment.
    """
    return coupon_term * principal


def annuity_installment(coupon_term: float,
                        principal: float,
                        _yield: float) -> float:
    """Installment payment for current loan principal.

    Args:
        coupon_term: Coupon rate per term.
        principal: Current loan principal.
        _yield: Annuity yield per term.

    Returns:
        Installment payment.
    """
    return _yield - annuity_interest(coupon_term, principal)


def annuity(coupon: float,
            frequency: int,
            cf_grid: np.ndarray,
            principal: float = 1) -> np.ndarray:
    """Cash flow of annuity.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        cf_grid: Grid of cash flow events. Assumed equidistant.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = cf_grid.size
    cash_flows = np.zeros((2, n_payments))
    # Coupon rate per term.
    coupon_term = coupon / frequency
    # Annuity yield.
    _yield = annuity_yield(n_payments, coupon_term, principal)
    # Remaining principal before payment.
    remaining_principal = principal
    for idx in range(n_payments):
        # Installment payment.
        cash_flows[0, idx] = \
            annuity_installment(coupon_term, remaining_principal, _yield)
        # Interest payment.
        cash_flows[1, idx] = annuity_interest(coupon_term, remaining_principal)
        # Subtract installment payment.
        remaining_principal -= cash_flows[0, idx]
    return cash_flows


def deferred_annuity(coupon: float,
                     frequency: int,
                     cf_grid: np.ndarray,
                     principal: float = 1,
                     io: int = 0) -> np.ndarray:
    """Cash flow of deferred annuity.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        cf_grid: Grid of cash flow events. Assumed equidistant.
        principal: Loan principal.
        io: Number of 'interest only' terms for deferred annuities.
            Default is 0.

    Returns:
        Cash flow.
    """
    # Number of interest payments.
    n_payments = cf_grid.size
    cash_flows = np.zeros((2, n_payments))
    # Coupon rate per term.
    coupon_term = coupon / frequency
    # Number of installment payments.
    n_install = n_payments - io
    # Annuity yield.
    _yield = annuity_yield(n_install, coupon_term, principal)
    # Interest payments in initial 'interest only' period.
    cash_flows[1, :io] = coupon_term * principal
    # Remaining principal before payment.
    remaining_principal = principal
    for idx in range(io, n_payments):
        # Installment payment.
        cash_flows[0, idx] = \
            annuity_installment(coupon_term, remaining_principal, _yield)
        # Interest payment.
        cash_flows[1, idx] = annuity_interest(coupon_term, remaining_principal)
        # Subtract installment payment.
        remaining_principal -= cash_flows[0, idx]
    return cash_flows


def standing_loan(coupon: float,
                  frequency: int,
                  cf_grid: np.ndarray,
                  principal: float = 1) -> np.ndarray:
    """Cash flow of standing loan.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        cf_grid: Grid of payment dates. Assumed equidistant.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = cf_grid.size
    cash_flows = np.zeros((2, n_payments))
    # Constant interest payment per term.
    cash_flows[1, :] = coupon * principal / frequency
    # Principal payed at maturity.
    cash_flows[0, -1] = principal
    return cash_flows


def serial_loan(coupon: float,
                frequency: int,
                cf_grid: np.ndarray,
                principal: float = 1) -> np.ndarray:
    """Cash flow of serial loan.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        cf_grid: Grid of cash flow events. Assumed equidistant.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = cf_grid.size
    cash_flows = np.zeros((2, n_payments))
    # Constant installment payment per term.
    cash_flows[0, :] = principal / n_payments
    # Coupon rate per term.
    coupon_term = coupon / frequency
    # Interest payment at each term.
    for idx in range(n_payments):
        cash_flows[1, idx] = coupon_term * cash_flows[0, idx:].sum()
    return cash_flows


def print_cash_flow(cf: np.ndarray):
    """Print cash flow to screen.

    Args:
        cf: Cash flow.
    """
    if len(cf.shape) == 1:
        for idx, total in enumerate(cf):
            print(f"Term {idx + 1:4}:  "
                  f"Total = {total:10.5f}")
        print(f"Summation:  "
              f"Total = {cf.sum():10.5f}")
    else:
        for idx, (installment, interest) in enumerate(zip(cf[0, :], cf[1, :])):
            print(f"Term {idx + 1:4}:  "
                  f"Installment = {installment:10.5f},  "
                  f"Interest = {interest:10.5f},  "
                  f"Total = {installment + interest:10.5f}")
        print(f"Summation:  "
              f"Installment = {cf[0, :].sum():10.5f},  "
              f"Interest = {cf[1, :].sum():10.5f},  "
              f"Total = {cf.sum():10.5f}")
