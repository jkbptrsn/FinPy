import math
import typing

import numpy as np

from models import options
from models.hull_white import misc as misc_hw
from models.hull_white import zero_coupon_bond as zcbond
from utils import global_types
from utils import misc


class CapFloor(options.EuropeanOptionAnalytical1F):
    """Cap or floor in 1-factor Hull-White model.

    TODO: See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.2, and
     D. Brigo & F. Mercurio 2007, section 3.3.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        strike_rate: Cap or floor rate.
        fixing_schedule: Fixing indices on event grid.
        payment_schedule: Payment indices on event grid.
        cap_or_floor: Caplet or floorlet. Default is caplet.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 strike_rate: float,
                 fixing_schedule: np.ndarray,
                 payment_schedule: np.ndarray,
                 event_grid: np.ndarray,
                 cap_or_floor: str = "cap",
                 time_dependence: str = "piecewise",
                 int_step_size: float = 1 / 365):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.strike_rate = strike_rate
        self.fixing_schedule = fixing_schedule
        self.payment_schedule = payment_schedule
        self.event_grid = event_grid
        self.cap_or_floor = cap_or_floor
        self.time_dependence = time_dependence
        self.int_step_size = int_step_size

        # Speed of mean reversion on event grid.
        self.kappa_eg = None
        # Volatility on event grid.
        self.vol_eg = None
        # Discount curve on event grid.
        self.discount_curve_eg = None
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = None
        # y-function on event grid.
        self.y_eg = None
        # v-function on event grid.
        self.v_eg = None

        # Zero-coupon bond object used in analytical pricing.
        self.zcbond = \
            zcbond.ZCBondNew(kappa, vol, discount_curve, event_grid.size - 1,
                             event_grid, time_dependence, int_step_size)

        # self.initialization()

        self.model = global_types.Model.HULL_WHITE_1F
        if self.cap_or_floor == "cap":
            self.type = global_types.Instrument.CAP
        elif self.cap_or_floor == "floor":
            self.type = global_types.Instrument.FLOOR
        else:
            raise ValueError(f"Unknown instrument type: {self.cap_or_floor}")
