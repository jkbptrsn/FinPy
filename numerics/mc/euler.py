import numpy as np
from scipy.stats import norm


class Euler1D:
    """One-dimensional Euler scheme.

    The SDE is of the form
        dx_t = drift * dt + diffusion * dW_t.
    """

    def __init__(self,
                 integration_grid: np.ndarray,
                 event_grid: np.ndarray):
        self.integration_grid = integration_grid
        self.event_grid = event_grid

        self.drift = 1
        self.diffusion = 1

    def full_path(self,
                  spot: float) -> np.ndarray:
        n_steps = self.integration_grid.size - 1
        wiener = norm.rvs(n_steps)
        x = np.ndarray(self.integration_grid.size)
        x[0] = spot
        for idx, (dt, dw) in enumerate(zip(np.diff(self.integration_grid), wiener)):
            x[idx + 1] = self.drift * dt + self.diffusion * dw + x[idx]
        return x
