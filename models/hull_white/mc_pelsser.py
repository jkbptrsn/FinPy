import math

import numpy as np

from models.hull_white import mc_andersen as mc_a
from utils import data_types
from utils import misc


class SdeConstant(mc_a.SdeExactConstant):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is given by
        dx_t = -kappa * x_t * dt + vol * dW_t,
    where
        x_t = r_t - f(0,t) - alpha_t.

    The speed of mean reversion and the volatility strip are constant.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility strip.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 event_grid: np.ndarray):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         event_grid)

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0


class SdePiecewise(mc_a.SdeExactPiecewise):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is given by
        dx_t = -kappa * x_t * dt + vol * dW_t,
    where
        x_t = r_t - f(0,t) - alpha_t.

    The speed of mean reversion is constant and the volatility strip is
    piecewise constant.

    TODO: Implicit assumption that all vol-strip events are represented on the event grid.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility strip.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 event_grid: np.ndarray):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         event_grid)

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0


class SdeGeneral(mc_a.SdeExactGeneral):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is given by
        dx_t = -kappa * x_t * dt + vol * dW_t,
    where
        x_t = r_t - f(0,t) - alpha_t.

    No assumption on the time-dependence of the speed of mean reversion
    and the volatility strip.

    TODO: Implicit assumption that all vol-strip events are represented on the event grid.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility strip.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         event_grid,
                         int_step_size)

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0
