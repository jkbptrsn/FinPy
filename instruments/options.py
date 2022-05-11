import abc


class VanillaOption(metaclass=abc.ABCMeta):
    """Abstract vanilla option class."""

    @property
    @abc.abstractmethod
    def strike(self) -> float:
        pass

    @strike.setter
    @abc.abstractmethod
    def strike(self, strike_):
        pass

    @property
    @abc.abstractmethod
    def expiry(self) -> float:
        pass

    @expiry.setter
    @abc.abstractmethod
    def expiry(self, expiry_):
        pass
