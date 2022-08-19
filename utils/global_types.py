from enum import Enum


class InstrumentType(Enum):
    AMERICAN_CALL = 1
    AMERICAN_PUT = 2
    BINARY_CASH_CALL = 3
    BINARY_ASSET_CALL = 4
    BINARY_CASH_PUT = 5
    BINARY_ASSET_PUT = 6
    EUROPEAN_CALL = 7
    EUROPEAN_PUT = 8
    SWAP = 9
    ZERO_COUPON_BOND = 10


class ModelName(Enum):
    BACHELIER = 1
    BLACK_SCHOLES = 2
    CIR = 3
    HULL_WHITE_1F = 4
    VASICEK = 5


class OptionExerciseType(Enum):
    AMERICAN = 1
    EUROPEAN = 2
