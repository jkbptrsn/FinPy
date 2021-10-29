import abc


class AbstractSDE(metaclass=abc.ABCMeta):
    """
    Abstract SDE class
    """
    @property
    @abc.abstractmethod
    def model_name(self):
        pass
