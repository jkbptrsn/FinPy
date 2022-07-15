import math


def a_factor(time1: float,
             time2: float,
             kappa: float,
             mean_rate: float,
             vol: float) -> float:
    """Eq. (3.25), Brigo & Mercurio 2007."""
    h = math.sqrt(kappa ** 2 + 2 * vol ** 2)
    exp_kappa_h = math.exp((kappa + h) * (time2 - time1) / 2)
    exp_h = math.exp(h * (time2 - time1))
    exponent = 2 * kappa * mean_rate / vol ** 2
    return exponent \
        * math.log(2 * h * exp_kappa_h / (2 * h + (kappa + h) * (exp_h - 1)))


def b_factor(time1: float,
             time2: float,
             kappa: float,
             vol: float) -> float:
    """Eq. (3.25), Brigo & Mercurio 2007."""
    h = math.sqrt(kappa ** 2 + 2 * vol ** 2)
    exp_h = math.exp(h * (time2 - time1))
    return 2 * (exp_h - 1) / (2 * h + (kappa + h) * (exp_h - 1))


def dadt(time1: float,
         time2: float,
         kappa: float,
         mean_rate: float,
         vol: float) -> float:
    """Time derivative of A: Eq. (3.25), Brigo & Mercurio 2007."""
    h = math.sqrt(kappa ** 2 + 2 * vol ** 2)
    exp_kappa_h = math.exp((kappa + h) * (time2 - time1) / 2)
    exp_h = math.exp(h * (time2 - time1))
    exponent = 2 * kappa * mean_rate / vol ** 2
    return (exponent / math.exp(a_factor(time1, time2, kappa, mean_rate, vol) / exponent)) \
        * (- h * (kappa + h) * exp_kappa_h / (2 * h + (kappa + h) * (exp_h - 1))
           + 2 * h ** 2 * (kappa + h) * exp_h * exp_kappa_h / (2 * h + (kappa + h) * (exp_h - 1)) ** 2)


def dbdt(time1: float,
         time2: float,
         kappa: float,
         vol: float) -> float:
    """Time derivative of B: Eq. (3.25), Brigo & Mercurio 2007."""
    h = math.sqrt(kappa ** 2 + 2 * vol ** 2)
    exp_h = math.exp(h * (time2 - time1))
    return - 2 * h * exp_h / (2 * h + (kappa + h) * (exp_h - 1)) \
        + 2 * (exp_h - 1) * h * (kappa + h) * exp_h \
        / (2 * h + (kappa + h) * (exp_h - 1)) ** 2
