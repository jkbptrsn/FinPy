# finpy
Financial engineering library in Python

Models
- Equity
  - Bachelier: dS_t = c1 * dW_t
  - Black-Scholes:  dS_t / S_t = c1 * dt + c2 * dW_t
- Short rate
  - Vasicek: dr_t = c1 * (c2 - r_t) * dt + c3 * dW_t
  - Cox-Ingersoll-Ross: dr_t = c1 * (c2 - r_t) * dt + c3 * sqrt{r_t} * dW_t
  - Hull-White (Extended Vasicek): dr_t = c1_t * (c2_t - r_t) * dt + c3_t * dW_t
