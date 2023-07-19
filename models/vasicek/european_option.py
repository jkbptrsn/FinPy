import math
import typing

import numpy as np

from models import options
from models.vasicek import misc
from models.vasicek import sde
from models.vasicek import zero_coupon_bond as zcbond
from utils.global_types import Instrument
from utils.global_types import Model
from utils import payoffs


class EuropeanOption(options.Option1FAnalytical):
    """European call/put option in Vasicek model.

    European call/put option written on a zero-coupon bond.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        strike: Strike price of zero-coupon bond at expiry.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
        type_: Option type. Default is call.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 event_grid: np.ndarray,
                 type_: str = "Call"):
        super().__init__()
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid

        # Underlying zero-coupon bond.
        self.zcbond = zcbond.ZCBond(self.kappa, self.mean_rate, self.vol,
                                    self.maturity_idx, self.event_grid)

        self.model = Model.VASICEK
        if type_ == "Call":
            self.type = Instrument.EUROPEAN_CALL
        elif type_ == "Put":
            self.type = Instrument.EUROPEAN_PUT
        else:
            raise ValueError(f"Option type is unknown: {type_}")

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.zcbond.maturity

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot value of underlying zero-coupon bond.

        Returns:
            Payoff.
        """
        if self.type == Instrument.EUROPEAN_CALL:
            return payoffs.call(spot, self.strike)
        else:
            return payoffs.put(spot, self.strike)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return misc.european_option_price(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            self.type)

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        return misc.european_option_delta(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            self.type)

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        return misc.european_option_gamma(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            self.type)

    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        return misc.european_option_theta(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            self.type)

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = self.zcbond.payoff(self.fd.grid)
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for idx, dt in enumerate(time_steps):
            event_idx = self.event_grid.size - idx - 1
            # Expiry of option.
            if event_idx == self.expiry_idx:
                self.fd.solution = self.payoff(self.fd.solution)
            self.fd.propagation(dt)

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        self.mc_exact = \
            sde.SdeExact(self.kappa, self.mean_rate, self.vol, self.event_grid)

    def mc_exact_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Exact discretization.

        Args:
            spot: Spot short rate.
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
        # Short rates at expiry.
        rates = self.mc_exact.rate_paths[self.expiry_idx]
        # Zero-coupon bond prices.
        zcbond_prices = self.zcbond.price(rates, self.expiry_idx)
        # Option payoffs.
        option_prices = self.payoff(zcbond_prices)
        # Discounted payoffs.
        option_prices *= self.mc_exact.discount_paths[self.expiry_idx]
        self.mc_exact.mc_estimate = option_prices.mean()
        self.mc_exact.mc_error = option_prices.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self):
        """Setup Euler Monte-Carlo solver."""
        self.mc_euler = \
            sde.SdeEuler(self.kappa, self.mean_rate, self.vol, self.event_grid)

    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Spot short rate.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of short rate and discount processes
            represented on event grid.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        # Short rates at expiry.
        rates = self.mc_euler.rate_paths[self.expiry_idx]
        # Zero-coupon bond prices.
        zcbond_prices = self.zcbond.price(rates, self.expiry_idx)
        # Option payoffs.
        option_prices = self.payoff(zcbond_prices)
        # Discounted payoffs.
        option_prices *= self.mc_euler.discount_paths[self.expiry_idx]
        self.mc_euler.mc_estimate = option_prices.mean()
        self.mc_euler.mc_error = option_prices.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)
