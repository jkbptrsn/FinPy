# finpy
Financial engineering library in Python

Models
- Equity
  - Bachelier
  - Black-Scholes
- Short rate
  - Vasicek: dr_t = a * (b - r_t) * dt + c * dW_t
  - Hull-White (Extended Vasicek): dr_t = a_t * (b_t - r_t) * dt + c_t * dW_t
  - Cox-Ingersoll-Ross: dr_t = a * (b - r_t) * dt + c * \sqrt{r_t} * dW_t
