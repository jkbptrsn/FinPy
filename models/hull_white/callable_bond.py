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


class FixedRate(bonds.BondAnalytical1F):
    """Fixed rate callable bond in 1-factor Hull-White model.

    TODO:
     * wrapper for calculating OAS based on price.
     * price-rate plots.

    Fixed rate callable bond with pre-specified cash flow.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event grid.

        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        deadline_schedule: Deadline indices on event grid.
        cash_flow_schedule: Cash flow indices on event grid.
        cash_flow: Cash flow.

        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        _oas: Option-adjusted spread.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 coupon: float,
                 frequency: int,
                 deadline_schedule: np.ndarray,
                 cash_flow_schedule: np.ndarray,
                 cash_flow: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 oas: float = 0):
        super().__init__()
        self.kappa = kappa
        self.vol = vol
        self.discount_curve = discount_curve
        self.coupon = coupon
        self.frequency = frequency
        self.deadline_schedule = deadline_schedule
        self.cash_flow_schedule = cash_flow_schedule
        self.cash_flow = cash_flow
        self.event_grid = event_grid
        self.time_dependence = time_dependence
        self.int_dt = int_dt
        self._oas = oas

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBond(kappa,
                          vol,
                          discount_curve,
                          self.cash_flow_schedule[-1],
                          event_grid,
                          time_dependence,
                          int_dt)
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

        # Is the bond callable?
        self.callable_bond = True

    def maturity(self) -> float:
        return self.event_grid[self.cash_flow_schedule[-1]]

    @property
    def oas(self) -> float:
        return self._oas

    @oas.setter
    def oas(self, oas_in):
        self._oas = oas_in
        self.oas_discount_steps = np.exp(-self._oas * np.diff(self.event_grid))
        self.oas_discount_steps = np.append(1, self.oas_discount_steps)

        # TODO: Important to NOT overwrite -- will affect adjustment in self.zcbond.
        self.adjust_discount_steps = \
            self.adjust_discount_steps * self.oas_discount_steps
        self.adjust_discount = \
            self.adjust_discount * np.cumprod(self.oas_discount_steps)

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot pseudo short rate.

        Returns:
            Payoff.
        """
        return self.cash_flow[:, -1].sum() + 0 * spot

    def price(self,
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
        for counter, idx_pay in enumerate(self.cash_flow_schedule):
            # Discount factor.
            discount = self.zcbond_price(spot, event_idx, idx_pay)
            _price += discount * self.cash_flow[:, counter].sum()
        return _price

    def delta(self,
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
        for counter, idx_pay in enumerate(self.cash_flow_schedule):
            # 1st order derivative of discount factor wrt short rate.
            discount = self.zcbond_delta(spot, event_idx, idx_pay)
            _delta += discount * self.cash_flow[:, counter].sum()
        return _delta

    def gamma(self,
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
        for counter, idx_pay in enumerate(self.cash_flow_schedule):
            # 2nd order derivative of discount factor wrt short rate.
            discount = self.zcbond_gamma(spot, event_idx, idx_pay)
            _gamma += discount * self.cash_flow[:, counter].sum()
        return _gamma

    def theta(self,
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
        for counter, idx_pay in enumerate(self.cash_flow_schedule):
            # 1st order derivative of discount factor wrt short rate.
            discount = self.zcbond_theta(spot, event_idx, idx_pay)
            _theta += discount * self.cash_flow[:, counter].sum()
        return _theta

    ####################################################################

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = np.zeros(self.fd.grid.size)
        # Update drift, diffusion and rate vectors.
        self.fd_update(self.event_grid.size - 1)
        # Backwards propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for counter, dt in enumerate(time_steps):
            event_idx = (self.event_grid.size - 1) - counter
            # Update drift, diffusion and rate vectors at previous event.
            self.fd_update(event_idx - 1)

            # # Evaluation based on cash flow events.
            # if event_idx in self.cash_flow_schedule:
            #     which_pay = np.where(self.cash_flow_schedule == event_idx)[0]
            #     idx_pay = which_pay[0]
            #     if self.callable_bond:
            #         # Constant prepayment rate.
            #         prepayment_rate = 0.2
            #         # Value of prepayment option
            #         option_value = -payoffs.call(self.fd.solution, 100)
            #         option_value = \
            #             smoothing.smoothing_1d(self.fd.grid, option_value)
            #         redemption_remaining = self.cash_flow[0, idx_pay:].sum()
            #         redemption_rate = \
            #             self.cash_flow[0, idx_pay] / redemption_remaining
            #         interest_rate = self.coupon / self.frequency
            #         self.fd.solution *= (1 - redemption_rate)
            #         self.fd.solution += 100 * (redemption_rate + interest_rate)
            #         self.fd.solution += prepayment_rate * option_value
            #     else:
            #         self.fd.solution += self.cash_flow[:, idx_pay].sum()

            # Evaluation based on deadline events.
            if event_idx in self.deadline_schedule:
                which_deadline = np.where(self.deadline_schedule == event_idx)[0]
                idx_deadline = which_deadline[0]
                idx_pay = idx_deadline

                # Present value of zero-coupon bond (unit notional) with
                # maturity at corresponding cash flow event.
                zcbond_mat_idx = self.cash_flow_schedule[idx_pay]
                zcbond_pv_tmp = \
                    self.zcbond_price(self.fd.grid,
                                      event_idx,
                                      zcbond_mat_idx)
                # OAS adjustment.
                slice_tmp = slice(event_idx, zcbond_mat_idx + 1)
                oas_adjustment = \
                    np.prod(self.oas_discount_steps[slice_tmp])
                zcbond_pv_tmp *= oas_adjustment

                if self.callable_bond:
                    # Constant prepayment rate.
                    prepayment_rate = 0.2
                    # Notional of 100.
                    zcbond_pv_tmp *= 100
                    # Value of prepayment option
                    option_value = \
                        payoffs.call(self.fd.solution, zcbond_pv_tmp)
                    option_value = \
                        smoothing.smoothing_1d(self.fd.grid, option_value)
                    redemption_remaining = self.cash_flow[0, idx_pay:].sum()
                    redemption_rate = \
                        self.cash_flow[0, idx_pay] / redemption_remaining
                    interest_rate = self.coupon / self.frequency
                    self.fd.solution *= (1 - redemption_rate)
                    self.fd.solution += \
                        (redemption_rate + interest_rate) * zcbond_pv_tmp
                    self.fd.solution -= prepayment_rate * option_value
                else:
                    self.fd.solution += \
                        self.cash_flow[:, idx_pay].sum() * zcbond_pv_tmp

            self.fd.propagation(dt, True)
            # Transformation adjustment.
            self.fd.solution *= self.adjust_discount_steps[event_idx]

        # If first deadline corresponds to first event.
        if self.deadline_schedule[0] == 0:
            # TODO: Same calculation as above. Create a new method?
            # TODO: The following statement is incorrect!.
            self.fd.solution += self.cash_flow[:, 0].sum()

    ####################################################################

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        self.zcbond.mc_exact_setup()
        self.mc_exact = self.zcbond.mc_exact

    def mc_exact_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of short rate and discount processes
            represented on event grid.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        present_value = self.mc_present_value(self.mc_exact)
        self.mc_exact.mc_estimate = present_value.mean()
        self.mc_exact.mc_error = present_value.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self):
        """Setup Euler Monte-Carlo solver."""
        self.zcbond.mc_euler_setup()
        self.mc_euler = self.zcbond.mc_euler

    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Monte-Carlo paths constructed using Euler-Maruyama discretization.

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

    ####################################################################

    def mc_present_value(self,
                         mc_object):
        """Present value for each Monte-Carlo path."""
        # Adjustment of discount paths.
        discount_paths = \
            mc_object.discount_adjustment(mc_object.discount_paths,
                                          self.adjust_discount)

        # TODO: Double check discounting...
        # Stepwise discount factors for all paths.
        discount_paths_steps = discount_paths[self.cash_flow_schedule, :]
        discount_paths_steps = \
            discount_paths_steps[1:, :] / discount_paths_steps[:-1, :]
        first_row = np.ndarray((1, discount_paths_steps.shape[1]))
        first_row[0, :] = discount_paths[self.cash_flow_schedule[0], :] \
            / discount_paths[0, :]
        discount_paths_steps = np.r_[first_row, discount_paths_steps]

        bond_payoff = np.zeros(mc_object.discount_paths.shape[1])

        # # Evaluation based on cash flow events.
        # for idx_pay in np.flip(self.cash_flow_schedule):
        #     which_pay = np.where(self.cash_flow_schedule == idx_pay)[0]
        #     counter = which_pay[0]
        #     if self.callable_bond:
        #         # Constant prepayment rate.
        #         prepayment_rate = 0.2
        #         # Value of prepayment option
        #         if idx_pay == self.cash_flow_schedule[-1]:
        #             option_value = np.zeros(mc_object.discount_paths.shape[1])
        #         else:
        #             # TODO: Not last cash flow event, but the first included, if not at time zero!
        #             # TODO: What about option value at time zero?
        #             option_value = \
        #                 lsm.prepayment_option(mc_object.rate_paths[idx_pay],
        #                                       bond_payoff)
        #         redemption_remaining = self.cash_flow[0, counter:].sum()
        #         redemption_rate = \
        #             self.cash_flow[0, counter] / redemption_remaining
        #         interest_rate = self.coupon / self.frequency
        #         bond_payoff *= (1 - redemption_rate)
        #         # Installment and interest payment.
        #         bond_payoff += 100 * (redemption_rate + interest_rate)
        #         # Prepayment option.
        #         bond_payoff += prepayment_rate * option_value
        #         # Stepwise discounting to previous cash flow event.
        #         bond_payoff *= discount_paths_steps[counter]
        #     else:
        #         bond_payoff += \
        #             self.cash_flow[:, counter].sum() * discount_paths[idx_pay]

        # TODO: Double check discounting...
        # Stepwise discount factors for all paths.
        discount_paths_steps = discount_paths[self.deadline_schedule, :]
        discount_paths_steps = \
            discount_paths_steps[1:, :] / discount_paths_steps[:-1, :]
        last_row = np.ndarray((1, discount_paths_steps.shape[1]))
        last_row[0, :] = discount_paths[self.cash_flow_schedule[-1], :] \
            / discount_paths[self.deadline_schedule[-1], :]
        discount_paths_steps = np.r_[discount_paths_steps, last_row]

        # Evaluation based on deadline events.
        for idx_deadline in np.flip(self.deadline_schedule):
            which_deadline = np.where(self.deadline_schedule == idx_deadline)[0]
            counter = which_deadline[0]
            idx_pay = self.cash_flow_schedule[counter]

            # Present value of zero-coupon bond (unit notional) with
            # maturity at corresponding cash flow event.
            # TODO: Really, discount factor along each path. OAS is included.
            zcbond_pv_tmp = discount_paths[idx_pay, :] \
                / discount_paths[idx_deadline, :]

            if self.callable_bond:
                # Constant prepayment rate.
                prepayment_rate = 0.2

                # Notional of 100.
                zcbond_pv_tmp *= 100

                # Stepwise discounting to previous cash flow event.
                bond_payoff *= discount_paths_steps[counter]

                # Value of prepayment option
                # TODO: What if the first deadline corresponds to the first event?
                option_value = \
                    lsm.prepayment_option(mc_object.rate_paths[idx_deadline],
                                          bond_payoff,
                                          zcbond_pv_tmp)

                redemption_remaining = self.cash_flow[0, counter:].sum()
                redemption_rate = \
                    self.cash_flow[0, counter] / redemption_remaining
                interest_rate = self.coupon / self.frequency
                bond_payoff *= (1 - redemption_rate)
                # Installment and interest payment.
                bond_payoff += \
                    (redemption_rate + interest_rate) * zcbond_pv_tmp
                # Prepayment option.
                bond_payoff -= prepayment_rate * option_value
            else:
                bond_payoff += \
                    self.cash_flow[:, counter].sum() \
                    * discount_paths[idx_pay] * zcbond_pv_tmp
        # Last discounting.
        bond_payoff *= discount_paths[self.deadline_schedule[0]]

        return bond_payoff

    ####################################################################

    def zcbond_price(self,
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

    def zcbond_delta(self,
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

    def zcbond_gamma(self,
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

    def zcbond_theta(self,
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
        discount_curve: Discount curve represented on event grid.

        coupon: Yearly coupon rate.
        frequency: Yearly payment frequency.
        deadline_schedule: Deadline indices on event grid.
        cash_flow_schedule: Cash flow indices on event grid.
        cash_flow: Cash flow.

        event_grid: Event dates as year fractions from as-of date.
        time_dependence: Time dependence of model parameters.
            "constant": kappa and vol are constant.
            "piecewise": kappa is constant and vol is piecewise
                constant.
            "general": General time dependence.
            Default is "piecewise".
        int_dt: Integration step size. Default is 1 / 52.
        _oas: Option-adjusted spread.
    """

    def __init__(self,
                 kappa: data_types.DiscreteFunc,
                 vol: data_types.DiscreteFunc,
                 discount_curve: data_types.DiscreteFunc,
                 coupon: float,
                 frequency: int,
                 deadline_schedule: np.ndarray,
                 cash_flow_schedule: np.ndarray,
                 cash_flow: np.ndarray,
                 event_grid: np.ndarray,
                 time_dependence: str = "piecewise",
                 int_dt: float = 1 / 52,
                 oas: float = 0):
        super().__init__(kappa,
                         vol,
                         discount_curve,
                         coupon,
                         frequency,
                         deadline_schedule,
                         cash_flow_schedule,
                         cash_flow,
                         event_grid,
                         time_dependence,
                         int_dt,
                         oas)

        # Zero-coupon bond.
        self.zcbond = \
            zcbond.ZCBondPelsser(kappa,
                                 vol,
                                 discount_curve,
                                 self.cash_flow_schedule[-1],
                                 event_grid,
                                 time_dependence,
                                 int_dt)

        self.transformation = self.zcbond.transformation

        self.adjust_rate = self.zcbond.adjust_rate
        self.adjust_discount_steps = self.zcbond.adjust_discount_steps
        self.adjust_discount = self.zcbond.adjust_discount
