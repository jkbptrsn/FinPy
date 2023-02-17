from enum import Enum, auto


class Instrument(Enum):
    """Type of financial instrument."""
    AMERICAN_CALL = auto()
    AMERICAN_PUT = auto()
    BINARY_ASSET_CALL = auto()
    BINARY_CASH_CALL = auto()
    BINARY_ASSET_PUT = auto()
    BINARY_CASH_PUT = auto()
    CAPLET = auto()
    EUROPEAN_CALL = auto()
    EUROPEAN_PUT = auto()
    FLOORLET = auto()
    SWAP = auto()
    SWAPTION = auto()
    ZERO_COUPON_BOND = auto()


class Model(Enum):
    """Type of underlying stochastic model."""
    BACHELIER = auto()
    BLACK_SCHOLES = auto()
    CIR = auto()
    HULL_WHITE_1F = auto()
    VASICEK = auto()


class OptionExercise(Enum):
    """Type of option exercise."""
    AMERICAN = auto()
    BERMUDAN = auto()
    EUROPEAN = auto()
