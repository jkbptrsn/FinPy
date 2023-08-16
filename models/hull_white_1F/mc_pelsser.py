import math
import typing

import numpy as np

from models.hull_white_1F import mc_andersen as mc_a
from utils import data_types
from utils import global_types


def rate_adjustment(rate_paths: np.ndarray,
                    adjustment: np.ndarray) -> np.ndarray:
    """Adjust pseudo rate paths.

    Assume that pseudo rate paths and discount curve are represented
    on identical event grids.

    Args:
        rate_paths: Pseudo short rate along Monte-Carlo paths.
        adjustment: Sum of instantaneous forward rate and alpha function
            on event grid.

    Returns:
        Actual short rate paths.
    """
    return (rate_paths.transpose() + adjustment).transpose()


def discount_adjustment(discount_paths: np.ndarray,
                        adjustment: np.ndarray) -> np.ndarray:
    """Adjust pseudo discount paths.

    Assume that pseudo discount paths and discount curve are
    represented on identical event grids.

    Args:
        discount_paths: Pseudo discount factor along Monte-Carlo
            paths.
        adjustment: Product of discount curve and exponentiation of
            time-integrated alpha function on event grid.

    Returns:
        Actual discount paths.
    """
    tmp = discount_paths.transpose() * adjustment
    return tmp.transpose()


class SdeExactConstant(mc_a.SdeExactConstant):
    """SDE for pseudo short rate process in 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See A. Pelsser 2000, chapter 5.

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

        self.transformation = global_types.Transformation.PELSSER

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0

    @staticmethod
    def rate_adjustment(rate_paths: np.ndarray,
                        adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo rate paths."""
        return rate_adjustment(rate_paths, adjustment)

    @staticmethod
    def discount_adjustment(discount_paths: np.ndarray,
                            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo discount paths."""
        return discount_adjustment(discount_paths, adjustment)


class SdeExactPiecewise(mc_a.SdeExactPiecewise):
    """SDE for pseudo short rate process in 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See A. Pelsser 2000, chapter 5.

    Monte-Carlo paths constructed using exact discretization.

    The speed of mean reversion is constant and the volatility is
    piecewise constant.

    TODO: Implicit assumption that all vol-strip events are represented
     on event grid.

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

        self.transformation = global_types.Transformation.PELSSER

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0

    @staticmethod
    def rate_adjustment(rate_paths: np.ndarray,
                        adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo rate paths."""
        return rate_adjustment(rate_paths, adjustment)

    @staticmethod
    def discount_adjustment(discount_paths: np.ndarray,
                            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo discount paths."""
        return discount_adjustment(discount_paths, adjustment)


class SdeExactGeneral(mc_a.SdeExactGeneral):
    """SDE for pseudo short rate process in 1-factor Hull-White model.


    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t) * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See A. Pelsser 2000, chapter 5.

    Monte-Carlo paths constructed using exact discretization.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    TODO: Implicit assumption that all vol-strip events are represented
     on event grid.

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

        self.transformation = global_types.Transformation.PELSSER

        self.rate_mean[:, 1] = 0
        self.discount_mean[:, 1] = 0

    @staticmethod
    def rate_adjustment(rate_paths: np.ndarray,
                        adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo rate paths."""
        return rate_adjustment(rate_paths, adjustment)

    @staticmethod
    def discount_adjustment(discount_paths: np.ndarray,
                            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo discount paths."""
        return discount_adjustment(discount_paths, adjustment)


class SdeEuler(mc_a.SdeEuler):
    """SDE for pseudo short rate process in 1-factor Hull-White model.

    The pseudo short rate is defined by
        dx_t = -kappa_t * x_t * dt + vol_t * dW_t,
    where kappa and mean_rate are the speed of mean reversion and mean
    reversion level, respectively, and vol denotes the volatility. W_t
    is a Brownian motion process under the risk-neutral measure Q.

    The pseudo short rate is related to the short rate by
        x_t = r_t - f(0,t) - alpha_t.

    See A. Pelsser 2000, chapter 5.

    Monte-Carlo paths constructed using Euler-Maruyama discretization.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         event_grid,
                         time_dependence,
                         int_dt)

        self.transformation = global_types.Transformation.ANDERSEN

    def _rate_increment(self,
                        spot: typing.Union[float, np.ndarray],
                        event_idx: int,
                        dt: float,
                        normal_rand: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Increment short rate process one time step.

        Args:
            spot: Short rate at event corresponding to event_idx.
            event_idx: Index on event grid.
            dt: Time step.
            normal_rand: Realizations of independent standard normal
                random variables.

        Returns:
            Increment of short rate process.
        """
        kappa = self.kappa_eg[event_idx]
        exp_kappa = math.exp(-kappa * dt)
        wiener_increment = math.sqrt(dt) * normal_rand
        rate_increment = exp_kappa * spot \
            + self.vol_eg[event_idx] * wiener_increment - spot
        return rate_increment

    @staticmethod
    def rate_adjustment(rate_paths: np.ndarray,
                        adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo rate paths."""
        return rate_adjustment(rate_paths, adjustment)

    @staticmethod
    def discount_adjustment(discount_paths: np.ndarray,
                            adjustment: np.ndarray) -> np.ndarray:
        """Adjust pseudo discount paths."""
        return discount_adjustment(discount_paths, adjustment)
