# finpy
Financial engineering library in Python

Models
- Equity
  - Bachelier
    - Scaled Brownian motion
    - dS(t) = vol * dW(t)
  - Black-Scholes 
    - Geometric Brownian motion
    - dS(t) / S(t) = rate * dt + vol * dW(t)
- Short rate
  - Vasicek
    - Gaussian mean reverting process 
    - dr(t) = kappa * [ theta - r(t) ] * dt + vol * dW(t)
  - Hull-White (Extended Vasicek)
    - Gaussian mean reverting process
    - dr(t) = kappa(t) * [ theta(t) - r(t) ] * dt + vol(t) * dW(t)
  - Cox-Ingersoll-Ross
    - dr(t) = kappa * [ theta - r(t) ] * dt + vol * r(t)^(1/2) * dW(t)

Development
- PEP 8 style guide
- Docstring format?

Miscellaneous
- Best resources for learning Python
  - Python Deep Dive 1-4 by Fred Baptiste (udemy courses)
