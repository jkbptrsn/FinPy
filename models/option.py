import abc


class VanillaOption(metaclass=abc.ABCMeta):
    """Abstract vanilla option class."""

    # MOVE option_type TO THIS ABC

    @property
    @abc.abstractmethod
    def strike(self):
        pass

    @strike.setter
    @abc.abstractmethod
    def strike(self, val):
        pass

    @property
    @abc.abstractmethod
    def expiry(self):
        pass

    @expiry.setter
    @abc.abstractmethod
    def expiry(self, val):
        pass
