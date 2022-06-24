import abc


class Bond(metaclass=abc.ABCMeta):
    """Abstract bond class."""

    @property
    @abc.abstractmethod
    def maturity(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def maturity_idx(self) -> int:
        pass

    @maturity_idx.setter
    @abc.abstractmethod
    def maturity_idx(self,
                     maturity_idx_: int):
        pass
