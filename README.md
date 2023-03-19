# FinPy
Financial engineering library for Python

The library covers the following models
- Equity
  - Bachelier
    - Brownian motion with drift
    - dS(t) = r * S(t) * dt + vol * dW(t)
  - Black-Scholes 
    - Geometric Brownian motion
    - dS(t) = rate * S(t) * dt + vol * S(t) * dW(t)
  - Heston
    - dS(t) = r * S(t) * dt + v(t)^(1/2) * S(t) * dW_S(t)
    - dv(t) = kappa * [ theta - v(t) ] * dt + vol * v(t)^(1/2) * dW_v(t)
    - dW_S(t) * dW_v(t) = rho * dt
  - SABR, D(t) = exp(-r * (T - t))
    - dS(t) = r * S(t) * dt + D(t) * a(t) * S(t)^b * dW_S(t)
    - da(t) = vol * a(t) * dW_a(t)
    - dW_S(t) * dW_a(t) = rho * dt
- Short rate, 1-factor
  - Vasicek
    - Mean reverting Gaussian process
    - dr(t) = kappa * [ theta - r(t) ] * dt + vol * dW(t)
  - Hull-White (Extended Vasicek)
    - Mean reverting Gaussian process
    - dr(t) = kappa(t) * [ theta(t) - r(t) ] * dt + vol(t) * dW(t)
  - Cox-Ingersoll-Ross
    - Mean reverting square root process
    - dr(t) = kappa * [ theta - r(t) ] * dt + vol * r(t)^(1/2) * dW(t)

Development guidelines and coding standards
- PEP 8 and Google Python Style Guide are followed rigorously (unless...)
- Type annotations and docstrings in Google format are mandatory
- Be kind to your future self, write unit tests
