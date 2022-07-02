import abc


class VanillaOption(metaclass=abc.ABCMeta):
    """Abstract vanilla option class."""

    @property
    @abc.abstractmethod
    def strike(self) -> float:
        pass

    @strike.setter
    @abc.abstractmethod
    def strike(self,
               strike_: float):
        pass

    @property
    @abc.abstractmethod
    def expiry(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def expiry_idx(self) -> int:
        pass

    @expiry_idx.setter
    @abc.abstractmethod
    def expiry_idx(self,
                   expiry_idx_: int):
        pass
