import enum


class Instrument(enum.Enum):
    """Type of financial instrument."""
    AMERICAN_CALL = enum.auto()
    AMERICAN_PUT = enum.auto()
    BINARY_ASSET_CALL = enum.auto()
    BINARY_CASH_CALL = enum.auto()
    BINARY_ASSET_PUT = enum.auto()
    BINARY_CASH_PUT = enum.auto()
    BOND = enum.auto()
    CAP = enum.auto()
    CAPLET = enum.auto()
    EUROPEAN_CALL = enum.auto()
    EUROPEAN_PUT = enum.auto()
    FLOOR = enum.auto()
    FLOORLET = enum.auto()
    SWAP = enum.auto()
    SWAPTION = enum.auto()
    ZERO_COUPON_BOND = enum.auto()


class Model(enum.Enum):
    """Type of underlying stochastic model."""
    BACHELIER = enum.auto()
    BLACK_SCHOLES = enum.auto()
    CIR = enum.auto()
    HESTON = enum.auto()
    HULL_WHITE_1F = enum.auto()
    SABR = enum.auto()
    VASICEK = enum.auto()


class Transformation(enum.Enum):
    """Short rate transformation in Hull-White model."""
    ANDERSEN = enum.auto()
    PELSSER = enum.auto()
