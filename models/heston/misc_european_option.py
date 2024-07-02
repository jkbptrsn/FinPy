import cmath


def alpha(j, k):
    return -k ** 2 / 2 - 1j * k / 2 + 1j * j * k


def beta(j, k, lambda_, eta, rho):
    return lambda_ - rho * eta * j - 1j * rho * eta * k


def gamma(eta):
    return eta ** 2 / 2


def discriminant(j, k, lambda_, eta, rho):
    alpha_ = alpha(j, k)
    beta_ = beta(j, k, lambda_, eta, rho)
    gamma_ = gamma(eta)
    return cmath.sqrt(beta_ ** 2 - 4 * alpha_ * gamma_)


def r_func(sign, j, k, lambda_, eta, rho):
    beta_ = beta(j, k, lambda_, eta, rho)
    gamma_ = gamma(eta)
    d = discriminant(j, k, lambda_, eta, rho)
    if sign == "plus":
        return (beta_ + d) / (2 * gamma_)
    elif sign == "minus":
        return (beta_ - d) / (2 * gamma_)
    else:
        raise ValueError("Unknown sign.")


def g_func(j, k, lambda_, eta, rho):
    r_minus = r_func("minus", j, k, lambda_, eta, rho)
    r_plus = r_func("plus", j, k, lambda_, eta, rho)
    return r_minus / r_plus


def d_func(j, k, lambda_, eta, rho, tau):
    d = discriminant(j, k, lambda_, eta, rho)
    g = g_func(j, k, lambda_, eta, rho)
    r_minus = r_func("minus", j, k, lambda_, eta, rho)
    return r_minus * (1 - cmath.exp(-d * tau)) / (1 - g * cmath.exp(-d * tau))


def c_func(j, k, lambda_, eta, rho, tau):
    d = discriminant(j, k, lambda_, eta, rho)
    g = g_func(j, k, lambda_, eta, rho)
    r_minus = r_func("minus", j, k, lambda_, eta, rho)
    gamma_ = gamma(eta)
    result = 1 - g * cmath.exp(-d * tau)
    result /= 1 - g
    return lambda_ * (r_minus * tau - cmath.log(result) / gamma_)


def probability(j, x, variance, lambda_, theta, eta, rho, tau):
    n_steps = 101
    k_max = 100
    step_size = k_max / (n_steps - 1)
    integral = 0
    for i in range(n_steps):
        k = step_size * (i + 0.5)
        c = c_func(j, k, lambda_, eta, rho, tau)
        d = d_func(j, k, lambda_, eta, rho, tau)
        integrand = cmath.exp(c * theta + d * variance + 1j * k * x) / (1j * k)
        integral += integrand.real * step_size
    return 0.5 + integral / cmath.pi
