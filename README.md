# finpy
Financial engineering library in Python

Models
- Equity models
  - Bachelier
  - Black-Scholes
- Short rate models
  - Vasicek, dr_t = a(b - r_t)dt + cdW_t
  - CIR (Cox-Ingersoll-Ross), dr_t = a(b - r_t)dt + c\sqrt{r_t}dW_t
  - Hull-White 1-factor
    - dr_t = (theta(t) - alpha r_t)dt + sigma(t)dW_t
  - Extended Vasicek model
    - dr_t = (theta(t) - alpha(t) r_t)dt + sigma(t)dW_t
  - SABR, stochastic volatility