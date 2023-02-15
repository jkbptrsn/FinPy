from enum import Enum


class Instrument(Enum):
    AMERICAN_CALL = 1
    AMERICAN_PUT = 2
    BINARY_ASSET_CALL = 3
    BINARY_CASH_CALL = 4
    BINARY_ASSET_PUT = 5
    BINARY_CASH_PUT = 6
    CAPLET = 7
    EUROPEAN_CALL = 8
    EUROPEAN_PUT = 9
    FLOORLET = 10
    SWAP = 11
    SWAPTION = 12
    ZERO_COUPON_BOND = 13


class Model(Enum):
    BACHELIER = 1
    BLACK_SCHOLES = 2
    CIR = 3
    HULL_WHITE_1F = 4
    VASICEK = 5


class OptionExerciseType(Enum):
    AMERICAN = 1
    BERMUDAN = 2
    EUROPEAN = 3
