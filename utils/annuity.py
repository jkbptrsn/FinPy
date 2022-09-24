import matplotlib.pyplot as plt
import numpy as np


def annuity_factor(n_terms: int,
                   coupon: float) -> float:
    """Calculate annuity factor ("alfahage").

    The terms are assumed positioned equidistantly on the time axis.

        \sum_{t = 1}^{n} 1 / (1 + r)^t = (1 - (1 + r)^{-n}) / r,
    using the sum formula
        \sum_{i = 0}^{n - 1} x^i = (1 - x^n) / (1 - x).
    """
    return (1 - (1 + coupon) ** (-n_terms)) / coupon


def prepayment(principal: float,
               n_terms: int,
               coupon: float) -> float:
    """Constant prepayment/yield of annuity."""
    return principal / annuity_factor(n_terms, coupon)


def interest_payment(principal: float,
                     coupon: float) -> float:
    """Interest payment per term for a given principal."""
    return principal * coupon


def amortization_payment(principal: float,
                         coupon: float,
                         yield_: float) -> float:
    """Amortization payment per term for a given principal."""
    return yield_ - interest_payment(principal, coupon)


def cash_flow_issue_date(principal: float,
                         coupon: float,
                         n_terms: int,
                         n_io_terms: int,
                         term_frequency: int,
                         *,
                         plot: bool = False) \
        -> tuple[np.ndarray, np.ndarray]:
    """Construct cash flow for a single issue date."""
    yield_ = prepayment(principal, n_terms - n_io_terms, coupon)
    p_remaining = principal
    terms = np.arange(1, n_terms + 1) / term_frequency
    cash_flow = np.zeros((2, n_terms))
    for j in range(n_terms):
        if j >= n_io_terms:
            # Amortization payment.
            cash_flow[0, j] = amortization_payment(p_remaining, coupon, yield_)
            # Interest payment.
            cash_flow[1, j] = interest_payment(p_remaining, coupon)
            # Remaining principal.
            p_remaining -= cash_flow[0, j]
        else:
            # Interest payment.
            cash_flow[1, j] = interest_payment(principal, coupon)
    if plot:
        plt.plot(terms, cash_flow[0, :], "ob",
                 markersize=4, label="Amortization")
        plt.plot(terms, cash_flow[1, :], "or",
                 markersize=4, label="Interest")
        plt.plot(terms, cash_flow.sum(axis=0), "ok",
                 markersize=4, label="Total prepayment")
        plt.xlabel("Years from end of issue period")
        plt.ylabel("Payments")
        plt.legend()
        plt.show()
    return terms, cash_flow


def cash_flow_issue_period(principal: float,
                           coupon: float,
                           n_terms: int,
                           n_io_terms: int,
                           n_issue_terms: int,
                           term_frequency: int,
                           *,
                           plot: bool = False) \
        -> tuple[np.ndarray, np.ndarray]:
    """..."""
    p_per_issue_date = principal / n_issue_terms
    _, cash_flow_tmp = cash_flow_issue_date(p_per_issue_date, coupon, n_terms,
                                            n_io_terms, term_frequency)
    terms_issue = np.arange(-n_issue_terms, 0) / term_frequency
    terms = np.arange(-n_issue_terms + 1, n_terms + 1) / term_frequency
    cash_flow = np.zeros((2, n_issue_terms + n_terms))
    for issue_idx in range(n_issue_terms):
        cash_flow[:, issue_idx:n_terms + issue_idx] += cash_flow_tmp
    if plot:
        plt.plot(terms_issue, 0 * terms_issue, "dg",
                 markersize=7, label="Issue period")
        plt.plot(terms, cash_flow[0, :], "ob",
                 markersize=3, label="Amortization")
        plt.plot(terms, cash_flow[1, :], "or",
                 markersize=3, label="Interest")
        plt.plot(terms, cash_flow.sum(axis=0), "ok",
                 markersize=3, label="Total prepayment")
        plt.xlabel("Years from end of issue period")
        plt.xticks(range(int(terms.min()), int(terms.max()) + 1))
        plt.ylabel("Payments")
        plt.legend()
        plt.show()
    return terms, cash_flow


if __name__ == "__main__":

    # Principal at time zero.
    principal = 100

    # Fixed yearly coupon.
    coupon_yearly = 0.04

    # Term frequency per year.
    term_frequency = 4

    # Fixed coupon per term.
    coupon_per_term = coupon_yearly / term_frequency

    # Number of remaining terms.
    n_terms = 120

    # Number of terms in bond issue period.
    n_issue_terms = 4 * 3 + 1

    # Number of remaining Interest-Only terms.
    n_io_terms = 4 * 10

    cash_flow_issue_period(principal, coupon_per_term, n_terms, n_io_terms,
                           n_issue_terms, term_frequency, plot=True)

    # Simple annuity with full amortization.
    terms, cash_flow = \
        cash_flow_issue_date(principal, coupon_per_term, n_terms, 0,
                             term_frequency, plot=False)
    prepayment_ = prepayment(principal, n_terms, coupon_per_term)
    remaining_principal = principal
    for n in range(n_terms):
        print(f"\nPayment date = {n + 1}")
        print(f"Principal before payment = {remaining_principal:8.4f}")
        print(f"Yield = {prepayment_:8.4f}")
        remaining_principal -= cash_flow[0, n]
        print(f"Amortization = {cash_flow[0, n]:8.4f}")
        print(f"Interest = {cash_flow[1, n]:8.4f}")
        print(f"Principal after payment = {remaining_principal:8.4f}")
