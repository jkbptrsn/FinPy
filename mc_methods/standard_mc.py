from scipy.stats import norm


class MC:
    def __init__(self, n_paths):
        self._n_paths = n_paths

    @property
    def n_paths(self):
        return self._n_paths

    @n_paths.setter
    def n_paths(self, n_paths_):
        self._n_paths = n_paths_
