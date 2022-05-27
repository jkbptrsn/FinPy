import abc


class Bond(metaclass=abc.ABCMeta):
    """Abstract bond class."""

    @property
    @abc.abstractmethod
    def maturity(self) -> float:
        pass

    @maturity.setter
    @abc.abstractmethod
    def maturity(self,
                 maturity_: float):
        pass
