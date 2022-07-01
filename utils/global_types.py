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
    BACHELIER = 1
    BLACK_SCHOLES = 2
    CIR = 3
    HULL_WHITE_1F = 4
    VASICEK = 5
