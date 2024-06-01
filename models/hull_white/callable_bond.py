import math
import typing

import numpy as np

from models import bonds
from models.hull_white import zero_coupon_bond as zcbond
from numerics.mc import lsm
from utils import data_types
from utils import global_types
from utils import payoffs
from utils import smoothing


class FixedRate(bonds.Bond1FAnalytical):
    """Fixed rate callable bond in 1-factor Hull-White model.

    Fixed rate callable bond with pre-specified cash flow.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        deadline_schedule: Deadline indices on event grid.
        payment_schedule: Payment indices on event grid.
        cash_flow: Cash flow on payment grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        oas: Option-adjusted spread.
        callable_bond: Is the bond callable? Default is True.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            coupon: float,
            frequency: int,
            deadline_schedule: np.ndarray,
            payment_schedule: np.ndarray,
            cash_flow: np.ndarray,
            event_grid: np.ndarray,
            time_dependence: str = "piecewise",
            int_dt: float = 1 / 52,
            oas: float = 0,
            callable_bond: bool = True):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.coupon = coupon
        self.frequency = frequency
        self.deadline_schedule = deadline_schedule
        self.payment_schedule = payment_schedule
        self.cash_flow = cash_flow
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt
        self.callable_bond = callable_bond

        # Zero-coupon bond.
        self.zcbond = zcbond.ZCBond(
            kappa, vol, discount_curve, self.payment_schedule[-1], event_grid,
            time_dependence, int_dt)

        # Kappa on event grid.
        self.kappa_eg = self.zcbond.kappa_eg
        # Vol on event grid.
        self.vol_eg = self.zcbond.vol_eg
        # Discount curve on event grid.
        self.discount_curve_eg = self.zcbond.discount_curve_eg
        # Instantaneous forward rate on event grid.
        self.forward_rate_eg = self.zcbond.forward_rate_eg
        # y-function on event grid.
        self.y_eg = self.zcbond.y_eg

        self.model = self.zcbond.model
        self.transformation = self.zcbond.transformation
        self.type = global_types.Instrument.BOND

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount

        # Discount adjustment including OAS.
        self.oas = oas

    def maturity(self) -> float:
        return self.event_grid[self.payment_schedule[-1]]

    @property
    def oas(self) -> float:
        return self._oas

    @oas.setter
    def oas(self, oas_in: float) -> None:
        self._oas = oas_in
        self.oas_discount_steps = np.exp(-self._oas * np.diff(self.event_grid))
        self.oas_discount_steps = np.append(1, self.oas_discount_steps)
        # Do NOT overwrite -- will affect adjustment in self.zcbond.
        self.adjust_discount_steps = \
            self.adjust_discount_steps * self.oas_discount_steps
        # Do NOT overwrite -- will affect adjustment in self.zcbond.
        self.adjust_discount = \
            self.adjust_discount * np.cumprod(self.oas_discount_steps)

###############################################################################

    def oas_calc(
            self,
            marked_price: float,
            tolerance: float = 1.0e-5,
            oas_shift: float = 1.0) -> float:
        """Calculate OAS corresponding to marked_price.

        Args:
            marked_price: Observable marked price.
            tolerance: Newton-Raphson tolerance level.
            oas_shift: OAS step size in Newton-Raphson method.

        Returns:
            OAS estimate.
        """
        # Initial OAS guess.
        oas_guess = 0.0
        self.oas = oas_guess
        # Price according to OAS guess.
        self.fd_solve()
        price = self.fd.solution[(self.fd.grid.size - 1) // 2]
        # Newton-Raphson iteration.
        while abs(marked_price - price) > tolerance:
            # Shift OAS guess.
            self.oas = oas_guess + oas_shift
            # Price according to shifted OAS guess.
            self.fd_solve()
            price_tmp = self.fd.solution[(self.fd.grid.size - 1) // 2]
            # 1st order price sensitivity wrt OAS.
            d_price_d_oas = (price_tmp - price) / oas_shift
            # Update OAS guess.
            oas_guess -= price / d_price_d_oas
            self.oas = oas_guess
            # Price according to updated OAS guess.
            self.fd_solve()
            price = self.fd.solution[(self.fd.grid.size - 1) // 2]
        return oas_guess

###############################################################################

    def payoff(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot pseudo short rate.

        Returns:
            Payoff.
        """
        return self.cash_flow[:, -1].sum() + 0 * spot

    def price(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Bond price.
        """
        _price = 0
        for count, idx_pay in enumerate(self.payment_schedule):
            if event_idx <= idx_pay:
                # Discount factor.
                discount = self.zcbond_price(spot, event_idx, idx_pay)
                _price += discount * self.cash_flow[:, count].sum()
        return _price

    def delta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        _delta = 0
        for count, idx_pay in enumerate(self.payment_schedule):
            if event_idx <= idx_pay:
                # 1st order derivative of discount factor wrt short
                # rate.
                discount = self.zcbond_delta(spot, event_idx, idx_pay)
                _delta += discount * self.cash_flow[:, count].sum()
        return _delta

    def gamma(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        _gamma = 0
        for count, idx_pay in enumerate(self.payment_schedule):
            if event_idx <= idx_pay:
                # 2nd order derivative of discount factor wrt short
                # rate.
                discount = self.zcbond_gamma(spot, event_idx, idx_pay)
                _gamma += discount * self.cash_flow[:, count].sum()
        return _gamma

    def theta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        _theta = 0
        for count, idx_pay in enumerate(self.payment_schedule):
            if event_idx <= idx_pay:
                # 1st order derivative of discount factor wrt time.
                discount = self.zcbond_theta(spot, event_idx, idx_pay)
                _theta += discount * self.cash_flow[:, count].sum()
        return _theta

###############################################################################

    def fd_solve(self) -> None:
        """Run finite difference solver on event grid."""
        # Set terminal condition.
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Reset drift, diffusion and rate vectors at terminal event.
        self.fd_update(self.event_grid.size - 1)
        # Backwards propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - idx
            # Update drift, diffusion and rate vectors at previous
            # event.
            self.fd_update(event_idx - 1)

            # TODO: Check this!
            # TODO: What if the deadline corresponding to the first
            #  payment is in the past?
            if event_idx in self.deadline_schedule:
                self._fd_payment_evaluation(event_idx)

            # Propagation for one time step.
            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjust_discount_steps[event_idx]

        # TODO: Is this correct?
        if self.deadline_schedule[0] == 0:
            self._fd_payment_evaluation(0)

    def _fd_payment_evaluation(
            self,
            event_idx: int) -> None:
        """Evaluation of payment at deadline event.

        Args:
            event_idx: Index on event grid.
        """
        which_deadline = np.where(self.deadline_schedule == event_idx)[0]
        counter = which_deadline[0]
        # Present value of zero-coupon bond (unit notional) with
        # maturity at corresponding payment event.
        mat_idx = self.payment_schedule[counter]
        zcbond_pv_tmp = self.zcbond_price(self.fd.grid, event_idx, mat_idx)
        # OAS adjustment. TODO: Is the slicing correct?
        oas_adjustment = np.prod(self.oas_discount_steps[event_idx: mat_idx])
        zcbond_pv_tmp *= oas_adjustment
        if self.callable_bond:

            prepayment_rate = \
                prepayment_function(self.fd.grid, event_idx, self.zcbond)

            # Present value of bond with notional of 100.
            zcbond_pv_tmp *= 100
            # (Negative) Value of prepayment option.
            option_value = payoffs.call(self.fd.solution, zcbond_pv_tmp)
            # TODO: Check implementation of smoothing.
            option_value = smoothing.smoothing_1d(self.fd.grid, option_value)
            redemption_remaining = self.cash_flow[0, counter:].sum()
            redemption_rate = \
                self.cash_flow[0, counter] / redemption_remaining
            interest_rate = self.coupon / self.frequency
            # Adjust previous payments.
            self.fd.solution *= (1 - redemption_rate)
            # Current redemption and interest payments.
            self.fd.solution += \
                (redemption_rate + interest_rate) * zcbond_pv_tmp
            # Subtract value of prepayment option.
            self.fd.solution -= prepayment_rate * option_value
        else:
            self.fd.solution += \
                self.cash_flow[:, counter].sum() * zcbond_pv_tmp

###############################################################################

    def mc_exact_setup(self) -> None:
        """Setup exact Monte-Carlo solver."""
        self.zcbond.mc_exact_setup()
        self.mc_exact = self.zcbond.mc_exact

    def mc_exact_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Exact discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        present_value = self.mc_present_value(self.mc_exact)
        self.mc_exact.mc_estimate = present_value.mean()
        self.mc_exact.mc_error = present_value.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self) -> None:
        """Setup Euler Monte-Carlo solver."""
        self.zcbond.mc_euler_setup()
        self.mc_euler = self.zcbond.mc_euler

    def mc_euler_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        present_value = self.mc_present_value(self.mc_euler)
        self.mc_euler.mc_estimate = present_value.mean()
        self.mc_euler.mc_error = present_value.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)

###############################################################################

    def mc_present_value(
            self,
            mc_object) -> np.ndarray:
        """Present value for each Monte-Carlo path."""
        # Adjustment of discount paths.
        discount_paths = \
            mc_object.discount_adjustment(mc_object.discount_paths,
                                          self.adjust_discount)
        # Stepwise discount factors for all paths.
        discount_paths_steps = discount_paths[self.deadline_schedule, :]
        discount_paths_steps = \
            discount_paths_steps[1:, :] / discount_paths_steps[:-1, :]
        last_row = np.ndarray((1, discount_paths_steps.shape[1]))
        last_row[0, :] = discount_paths[self.payment_schedule[-1], :] \
            / discount_paths[self.deadline_schedule[-1], :]
        discount_paths_steps = np.r_[discount_paths_steps, last_row]
        bond_payoff = np.zeros(mc_object.discount_paths.shape[1])
        for idx_deadline in np.flip(self.deadline_schedule):
            which_deadline = \
                np.where(self.deadline_schedule == idx_deadline)[0]
            counter = which_deadline[0]
            idx_payment = self.payment_schedule[counter]
            # Path-dependent discount factor from previous payment event
            # to present deadline event. OAS is included.
            zcbond_pv_tmp = discount_paths[idx_payment, :] \
                / discount_paths[idx_deadline, :]
            if self.callable_bond:
                rate_paths = mc_object.rate_paths[idx_deadline]

                prepayment_rate = prepayment_function(
                    rate_paths, idx_deadline, self.zcbond)

                # Notional of 100 discounted along each path.
                zcbond_pv_tmp *= 100
                # Stepwise discounting from previous deadline event.
                bond_payoff *= discount_paths_steps[counter]
                # Value of prepayment option.
                if idx_deadline != 0:
                    option_value = lsm.prepayment_option(
                        rate_paths, bond_payoff, zcbond_pv_tmp)
                else:
                    bond_mean = bond_payoff.mean()
                    strike_mean = zcbond_pv_tmp.mean()
                    option_value = np.maximum(bond_mean - strike_mean, 0)
                redemption_remaining = self.cash_flow[0, counter:].sum()
                redemption_rate = \
                    self.cash_flow[0, counter] / redemption_remaining
                interest_rate = self.coupon / self.frequency
                # Adjust previous payments.
                bond_payoff *= (1 - redemption_rate)
                # Current redemption and interest payments.
                bond_payoff += \
                    (redemption_rate + interest_rate) * zcbond_pv_tmp
                # Subtract value of prepayment option.
                bond_payoff -= prepayment_rate * option_value
            else:
                bond_payoff += self.cash_flow[:, counter].sum() \
                    * discount_paths[idx_payment]
        if self.callable_bond:
            # Discounting from first deadline event to time zero.
            bond_payoff *= discount_paths[self.deadline_schedule[0]]
        return bond_payoff

###############################################################################

    def zcbond_price(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Price of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond price.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.price(spot, event_idx)

    def zcbond_delta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Delta of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond delta.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.delta(spot, event_idx)

    def zcbond_gamma(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Gamma of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond gamma.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.gamma(spot, event_idx)

    def zcbond_theta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int,
            maturity_idx: int) -> typing.Union[float, np.ndarray]:
        """Theta of zero-coupon bond.

        Args:
            spot: Spot pseudo short rate.
            event_idx: Index on event grid.
            maturity_idx: Maturity index on event grid.

        Returns:
            Zero-coupon bond theta.
        """
        if self.zcbond.mat_idx != maturity_idx:
            self.zcbond.mat_idx = maturity_idx
        return self.zcbond.theta(spot, event_idx)


class FixedRatePelsser(FixedRate):
    """Fixed rate callable Bond in 1-factor Hull-White model.

    Fixed rate callable bond with pre-specified cash flow.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve.
        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        deadline_schedule: Deadline indices on event grid.
        payment_schedule: Payment indices on event grid.
        cash_flow: Cash flow on payment grid.
        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            - "constant": kappa and vol are constant.
            - "piecewise": kappa is constant and vol is piecewise
                constant.
            - "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        oas: Option-adjusted spread.
        callable_bond: Is the bond callable? Default is True.
    """

    def __init__(
            self,
            kappa: data_types.DiscreteFunc,
            vol: data_types.DiscreteFunc,
            discount_curve: data_types.DiscreteFunc,
            coupon: float,
            frequency: int,
            deadline_schedule: np.ndarray,
            payment_schedule: np.ndarray,
            cash_flow: np.ndarray,
            event_grid: np.ndarray,
            time_dependence: str = "piecewise",
            int_dt: float = 1 / 52,
            oas: float = 0,
            callable_bond: bool = True):
        super().__init__(
            kappa, vol, discount_curve, coupon, frequency, deadline_schedule,
            payment_schedule, cash_flow, event_grid, time_dependence, int_dt,
            oas, callable_bond)

        # Zero-coupon bond.
        self.zcbond = zcbond.ZCBondPelsser(
            kappa, vol, discount_curve, self.payment_schedule[-1], event_grid,
            time_dependence, int_dt)

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount


###############################################################################


def prepayment_function(
        short_rate: (float, np.ndarray),
        event_idx: int,
        _zcbond: zcbond.ZCBond) -> (float, np.ndarray):
    """Prepayment function.

    Args:
        short_rate: Pseudo short rate(s).
        event_idx: Index on event grid.
        _zcbond: Zero-coupon bond object.

    Returns:
        Prepayment function.
    """
    # Constant prepayment rate.
    return 0.2
