import numpy as np

from models.hull_white import mc_andersen as mc_a
from utils import data_types


class SdeExactConstant(mc_a.SdeExactConstant):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See TODO: Pelsser ref.

    Monte-Carlo paths constructed using exact discretization.

    The speed of mean reversion and volatility is constant.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates as year fractions from as-of date.
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


class SdeExactPiecewise(mc_a.SdeExactPiecewise):
    """SDE class for 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See TODO: Pelsser ref.

    Monte-Carlo paths constructed using exact discretization.

    The speed of mean reversion is constant and the volatility is
    piecewise constant.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates as year fractions from as-of date.
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


class SdeExactGeneral(mc_a.SdeExactGeneral):
    """SDE class for 1-factor Hull-White model.


    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See TODO: Pelsser ref.

    Monte-Carlo paths constructed using exact discretization.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates as year fractions from as-of date.
        int_dt: Integration step size. Default is 1 / 52.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 event_grid: np.ndarray,
                 int_dt: float = 1 / 52):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         event_grid,
                         int_dt)

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0
