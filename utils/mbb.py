import numpy as np
import matplotlib.pyplot as plt

import utils.misc as misc


def bond_price(date, fondskode, spread, *, print_info=False):
    """Price bond."""
    return None


def bond_specs(date, fondskode, *, print_info=False):
    """Get bond specs."""
    return None


def borrower_distribution(date, fondskode, *, print_info=False):
    """Get borrower distribution."""
    pass


def cash_flow_obj(date, fondskode):
    """Get cash flow object."""
    return None


def cash_flow_grid(date, fondskode, *, print_info=False):
    """Term structure of cash flows on term/payment dates.

    The function also returns prepayment deadline dates. The day-count
    convention is ACT/365.2425.
    """
    return None


def cash_flow_plot(payment_grid, cash_flow):
    """Plot cash flow on payment grid."""
    pass


def date_obj(day, month, year):
    """Get date object."""
    return None


def refinance_spread(date, fondskode):
    """Spread to account for prepayment option of refinance bond."""
    return None


def prepayment_rate(date, term, deadline, fondskode, remaining_balance,
                    refi_coupon, repo_rate_in, zc10y_rate, zc10y_rate_diff,
                    *, extended=False):
    """Calculated prepayment rate."""
    return None


def volatility_strip(date, *, print_info=False):
    """Pull DKK volatility strip."""
    return None


def volatility_strip_plot(vol_strip):
    """Plot volatility strip."""
    pass


def zero_strip(date, event_grid=None, *, print_info=False):
    """Pull DKK swap curve."""
    return None


def zero_strip_plot(rate_curve):
    """Plot zero strip."""
    pass
