# finpy
Financial engineering library in Python

Models
- Equity
  - Bachelier: dS_t = vol * dW_t
  - Black-Scholes:  dS_t / S_t = rate * dt + vol * dW_t
- Short rate
  - Vasicek: dr_t = kappa * (theta - r_t) * dt + vol * dW_t
  - Cox-Ingersoll-Ross: dr_t = kappa * (theta - r_t) * dt + vol * sqrt{r_t} * dW_t
  - Hull-White (Extended Vasicek): dr_t = kappa_t * (theta_t - r_t) * dt + vol_t * dW_t
