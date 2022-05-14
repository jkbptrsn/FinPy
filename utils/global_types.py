from enum import Enum


class ExerciseType(Enum):
    EUROPEAN = 1
    AMERICAN = 2
    BERMUDAN = 3


class InstrumentType(Enum):
    EUROPEAN_CALL = 1
    EUROPEAN_PUT = 2
    AMERICAN_CALL = 3
    AMERICAN_PUT = 4
    BINARY_CASH_CALL = 5
    BINARY_ASSET_CALL = 6
    BINARY_CASH_PUT = 7
    BINARY_ASSET_PUT = 8
    ZERO_COUPON_BOND = 9


class ModelName(Enum):
    BLACk_SCHOLES = 1
    BACHELIER = 2
    VASICEK = 3
    CIR = 4
    HULL_WHITE = 5
