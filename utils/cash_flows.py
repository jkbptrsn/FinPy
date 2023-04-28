import math

import numpy as np


def cash_flow(coupon: float,
              frequency: int,
              payment_grid: np.ndarray,
              principal: float = 1,
              _type: str = "annuity") -> np.ndarray:
    """Cash flow.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment dates.
        principal: Loan principal.
        _type: Type of cash flow. annuity, standing or serial.
            Default is annuity.

    Returns:
        Cash flow.
    """
    cf = cash_flow_split(coupon, frequency, payment_grid, principal, _type)
    return cf.sum(axis=0)


def cash_flow_split(coupon: float,
                    frequency: int,
                    payment_grid: np.ndarray,
                    principal: float = 1,
                    _type: str = "annuity") -> np.ndarray:
    """Cash flows for both installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment dates.
        principal: Loan principal.
        _type: Type of cash flow. annuity, standing or serial.
            Default is annuity.

    Returns:
        Cash flows.
    """
    if _type == "annuity":
        return annuity(coupon, frequency, payment_grid, principal)
    elif _type == "standing":
        return standing_loan(coupon, frequency, payment_grid, principal)
    elif _type == "serial":
        return serial_loan(coupon, frequency, payment_grid, principal)
    else:
        raise ValueError(f"Cash flow type is unknown: {_type}")


def set_payment_grid(t_initial: float,
                     t_final: float,
                     frequency: int) -> np.ndarray:
    """Set up grid of payment dates.

    Args:
        t_initial: Initial time of first payment period.
        t_final: Final time of last payment period.
        frequency: Yearly payment frequency.

    Returns:
        Grid of payment dates.
    """
    # Year-faction between payment dates.
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


def annuity_interest(coupon: float,
                     principal: float) -> float:
    """Interest payment for current loan principal.

    Args:
        coupon: Coupon rate per term.
        principal: Current loan principal.

    Returns:
        Interest payment.
    """
    return coupon * principal


def annuity_installment(coupon: float,
                        principal: float,
                        _yield: float) -> float:
    """Installment payment for current loan principal.

    Args:
        coupon: Coupon rate per term.
        principal: Current loan principal.
        _yield: Annuity yield per term.

    Returns:
        Installment payment.
    """
    return _yield - annuity_interest(coupon, principal)


def annuity(coupon: float,
            frequency: int,
            payment_grid: np.ndarray,
            principal: float = 1) -> np.ndarray:
    """Cash flow of annuity.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment dates.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = payment_grid.size
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


def standing_loan(coupon: float,
                  frequency: int,
                  payment_grid: np.ndarray,
                  principal: float = 1) -> np.ndarray:
    """Cash flow of standing loan.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment dates.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = payment_grid.size
    cash_flows = np.zeros((2, n_payments))
    # Constant interest payment per term.
    cash_flows[1, :] = coupon * principal / frequency
    # Principal payed at maturity.
    cash_flows[0, -1] = principal
    return cash_flows


def serial_loan(coupon: float,
                frequency: int,
                payment_grid: np.ndarray,
                principal: float = 1) -> np.ndarray:
    """Cash flow of serial loan.

    Cash flow is split in installment and interest payments.

    Args:
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        payment_grid: Grid of payment dates.
        principal: Loan principal.

    Returns:
        Cash flow.
    """
    # Number of payments.
    n_payments = payment_grid.size
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
                  f"Total = {total}:10.5f")
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
